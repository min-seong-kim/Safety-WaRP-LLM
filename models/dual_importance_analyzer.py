"""
Dual Importance Analysis in WaRP Basis-Rotated Space

목적
----
Phase 3에서 "preserve(안전) mask로 freeze된 파라미터"가
"adapt(downstream) task에서도 중요한 파라미터"와 얼마나 겹치는지 정량화한다.

- 겹침이 크다  → Phase 3 FT에서 freeze가 downstream 성능을 방해함
- 겹침이 작다  → freeze해도 downstream task가 다른 방향을 자유롭게 쓸 수 있음

수식
----
importance_score[i,j] = Σ_batch |∂L/∂basis_coeff[i,j]|

여기서 basis_coeff = W @ U (Phase 1에서 구한 rotation)

preserve mask P : importance_preserve 상위 keep_ratio
adapt    mask A : importance_adapt    상위 keep_ratio

레이어별 overlap 지표:
  intersection        = |P ∩ A|
  union               = |P ∪ A|
  jaccard             = |P ∩ A| / |P ∪ A|
  adapt_blocked_ratio = |P ∩ A| / |A|      (adapt 중요 파라미터 중 preserve가 막는 비율)
  preserve_also_adapt = |P ∩ A| / |P|      (preserve freeze 중 adapt에도 중요한 비율)
  pearson_r           = 두 importance score의 Pearson 상관계수

출력
----
  analysis_<timestamp>/
    metadata.json
    importances_preserve.pt        (dict: key → np.ndarray)
    importances_adapt.pt           (dict: key → np.ndarray)
    masks_preserve/                (phase2와 동일 형식, 저장 옵션)
    masks_adapt/
    overlap_stats.json             (레이어별 + 전체 집계)
    overlap_stats.csv
    figures/
      overlap_by_layer.png         (레이어별 jaccard / adapt_blocked / preserve_also_adapt 막대)
      scatter_<layer>_<type>.png   (선택, --save_scatter)
      scatter_global.png           (전체 산점도)
      heatmap_jaccard.png          (레이어 × layer_type 히트맵)
"""

import gc
import json
import os
import random
import re
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

from .warp_modules import WaRPModule, switch_to_warp_module


# ---------------------------------------------------------------------------
# 메인 클래스
# ---------------------------------------------------------------------------

class DualImportanceAnalyzer:
    """
    WaRP basis-rotated space에서 두 데이터셋의 importance mask 겹침을 분석한다.

    Parameters
    ----------
    args               : argparse.Namespace (아래 속성 참고)
    logger             : logging.Logger
    basis_dir          : Phase 1 basis 디렉토리 경로
    phase0_model_dir   : Phase 0/base 모델 경로

    필수 args 속성
    --------------
    layer_type         : 'attn_q,attn_k,...'
    target_layers      : 'all' | '0-5' | '30'
    dtype              : 'bfloat16' | 'float16' | 'float32'
    device             : 'cuda' | 'cpu'
    batch_size         : int
    max_length         : int
    keep_ratio         : float  (0.1 = 상위 10% → mask=1)
    output_dir         : str
    preserve_dataset   : 'circuit_breakers'
    adapt_dataset      : 'gsm8k' | 'math' | 'metamath' | 'circuit_breakers' | 'wikipedia'
    preserve_samples   : int (0 = 전체)
    adapt_samples      : int (0 = 전체)
    circuit_breakers_path : str
    no_plot            : bool
    save_scatter       : bool  (per-layer scatter 저장 여부, 느릴 수 있음)
    """

    _LAYER_TYPE_TO_ATTR = {
        'attn_q':   ('self_attn', 'q_proj'),
        'attn_k':   ('self_attn', 'k_proj'),
        'attn_v':   ('self_attn', 'v_proj'),
        'attn_o':   ('self_attn', 'o_proj'),
        'ffn_gate': ('mlp',       'gate_proj'),
        'ffn_up':   ('mlp',       'up_proj'),
        'ffn_down': ('mlp',       'down_proj'),
    }

    def __init__(self, args, logger, basis_dir: str, phase0_model_dir: str):
        self.args             = args
        self.logger           = logger
        self.basis_dir        = basis_dir
        self.phase0_model_dir = phase0_model_dir

        self.model     = None
        self.tokenizer = None

        self.basis_data: Dict = {}
        self.layer_types      = []

        # raw importance (np.ndarray, float32) per key
        self.imp_preserve: Dict[Tuple, np.ndarray] = {}
        self.imp_adapt:    Dict[Tuple, np.ndarray] = {}

        # bool masks
        self.mask_preserve: Dict[Tuple, np.ndarray] = {}
        self.mask_adapt:    Dict[Tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _is_instruct_model(self) -> bool:
        return 'instruct' in str(self.phase0_model_dir).lower()

    def _parse_target_layers(self, num_layers: int):
        target = self.args.target_layers.strip()
        if target == 'all':
            return list(range(num_layers))
        if '-' in target:
            start, end = map(int, target.split('-'))
            return list(range(start, end + 1))
        return [int(target)]

    def _format_qa(self, question: str, answer: str) -> Tuple[str, str]:
        """(prompt_text, full_text) 반환 (instruct 모델에서 chat template 적용)."""
        question = str(question).strip()
        answer   = str(answer).strip()
        if self._is_instruct_model():
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False, add_generation_prompt=True,
                )
                full_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
                    tokenize=False, add_generation_prompt=False,
                )
                return prompt_text, full_text
            except Exception:
                pass
        prompt_text = f"Question: {question}\nAnswer:"
        full_text   = f"Question: {question}\nAnswer: {answer}"
        return prompt_text, full_text

    def _tokenize_qa(self, question: str, answer: str, max_length: int) -> dict:
        """SFT 형식 토크나이제이션 — prompt 부분 labels=-100."""
        prompt_text, full_text = self._format_qa(question, answer)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                    truncation=True, max_length=max_length)["input_ids"]
        full_ids   = self.tokenizer(full_text,   add_special_tokens=False,
                                    truncation=True, max_length=max_length)["input_ids"]
        labels = full_ids.copy()
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }

    @staticmethod
    def _pad_collate(batch, pad_id: int):
        max_len = max(len(f["input_ids"]) for f in batch)
        inp, attn, lbl = [], [], []
        for f in batch:
            pl = max_len - len(f["input_ids"])
            inp.append(f["input_ids"]      + [pad_id] * pl)
            attn.append(f["attention_mask"] + [0]     * pl)
            has_labels = "labels" in f
            if has_labels:
                lbl.append(f["labels"] + [-100] * pl)
        out = {
            "input_ids":      torch.tensor(inp,  dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        if lbl:
            out["labels"] = torch.tensor(lbl, dtype=torch.long)
        return out

    # ------------------------------------------------------------------
    # 1. Basis 로드
    # ------------------------------------------------------------------

    def load_basis(self):
        self.logger.info(f"Loading basis from {self.basis_dir} ...")
        metadata_path = os.path.join(self.basis_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            json.load(f)   # 검증 목적

        layer_types_str = self.args.layer_type
        self.layer_types = [lt.strip() for lt in layer_types_str.split(',')]

        total = 0
        for layer_type in self.layer_types:
            lt_dir = os.path.join(self.basis_dir, layer_type)
            if not os.path.exists(lt_dir):
                self.logger.warning(f"  basis dir not found: {lt_dir}")
                continue
            for svd_file in sorted(f for f in os.listdir(lt_dir)
                                   if f.startswith('layer_') and f.endswith('_svd.pt')):
                layer_idx = int(svd_file.split('_')[1])
                svd_data  = torch.load(os.path.join(lt_dir, svd_file), weights_only=False)
                self.basis_data[(layer_idx, layer_type)] = {'U': svd_data['U']}
                total += 1

        self.logger.info(f"✓ Basis loaded: {total} (layer, type) combinations")

    # ------------------------------------------------------------------
    # 2. 모델 로드 + WaRP 변환 + 재파라미터화
    # ------------------------------------------------------------------

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        torch_dtype = dtype_map.get(getattr(self.args, 'dtype', 'bfloat16'), torch.bfloat16)

        self.logger.info(f"Loading model from {self.phase0_model_dir} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.phase0_model_dir,
            torch_dtype=torch_dtype,
            device_map=getattr(self.args, 'device', 'cuda'),
            trust_remote_code=True,
        )
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.phase0_model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Converting to WaRP modules ...")
        self.model = switch_to_warp_module(self.model, self.layer_types, self.args.target_layers)

        # basis_coeff 공간으로 재파라미터화
        self.logger.info("Reparameterizing weights to basis_coeff space ...")
        n_reparam = 0
        target_indices = self._parse_target_layers(len(self.model.model.layers))
        for layer_idx in target_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                key = (layer_idx, layer_type)
                if key not in self.basis_data:
                    continue
                mod = self._get_warp_module(layer, layer_type)
                if mod is None:
                    continue

                W = mod.weight.data.clone()
                U = self.basis_data[key]['U'].to(dtype=W.dtype, device=W.device)

                mod.basis_coeff.data = W @ U
                mod.UT_forward  = U.clone().detach()
                mod.UT_backward = torch.empty(0, dtype=W.dtype, device=W.device)
                mod.flag        = True
                mod.coeff_mask.data.zero_()
                if hasattr(mod, 'mask_mode'):
                    mod.mask_mode.fill_(1)
                n_reparam += 1

        self.logger.info(f"✓ Reparameterized {n_reparam} modules")

    def _get_warp_module(self, layer, layer_type) -> Optional[WaRPModule]:
        if layer_type not in self._LAYER_TYPE_TO_ATTR:
            return None
        sub_name, proj_name = self._LAYER_TYPE_TO_ATTR[layer_type]
        try:
            mod = getattr(getattr(layer, sub_name), proj_name)
        except AttributeError:
            return None
        return mod if isinstance(mod, WaRPModule) else None

    # ------------------------------------------------------------------
    # 3. 데이터로더 빌더
    # ------------------------------------------------------------------

    def _build_dataloader(self, dataset_name: str, max_samples: int, seed: int = 112) -> DataLoader:
        """
        dataset_name : 'circuit_breakers' | 'gsm8k' | 'math' | 'metamath' | 'wikipedia'
        max_samples  : 0 = 전체
        """
        self.logger.info(f"  Building dataloader: {dataset_name} (max_samples={max_samples})")
        max_length = getattr(self.args, 'max_length', 1024)
        pad_id     = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def _collate(batch):
            return self._pad_collate(batch, pad_id)

        if dataset_name == 'circuit_breakers':
            return self._dl_circuit_breakers(max_samples, max_length, _collate, seed)
        if dataset_name == 'gsm8k':
            return self._dl_gsm8k(max_samples, max_length, _collate)
        if dataset_name == 'math':
            return self._dl_math(max_samples, max_length, _collate)
        if dataset_name == 'metamath':
            return self._dl_metamath(max_samples, max_length, _collate)
        if dataset_name == 'wikipedia':
            return self._dl_wikipedia(max_samples, max_length, _collate, seed)
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         "Choose from circuit_breakers, gsm8k, math, metamath, wikipedia")

    # ── circuit_breakers ──────────────────────────────────────────────

    def _dl_circuit_breakers(self, max_samples, max_length, collate_fn, seed):
        path = getattr(self.args, 'circuit_breakers_path', './data/circuit_breakers_train.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if max_samples > 0:
            data = data[:max_samples]

        tokenized = []
        for item in data:
            tokenized.append(self._tokenize_qa(
                item.get('prompt', ''),
                item.get('llama3_output', ''),
                max_length,
            ))

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(tokenized)
            def __getitem__(self, i): return tokenized[i]

        return DataLoader(_DS(), batch_size=self.args.batch_size, shuffle=False,
                          collate_fn=collate_fn,
                          generator=torch.Generator().manual_seed(seed))

    # ── GSM8K ─────────────────────────────────────────────────────────

    def _dl_gsm8k(self, max_samples, max_length, collate_fn):
        if hf_load_dataset is None:
            raise ImportError("datasets 라이브러리가 필요합니다: pip install datasets")
        raw = hf_load_dataset('openai/gsm8k', 'main', split='train')
        if max_samples > 0:
            raw = raw.select(range(min(max_samples, len(raw))))

        tokenized = [self._tokenize_qa(ex['question'], ex['answer'], max_length) for ex in raw]

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(tokenized)
            def __getitem__(self, i): return tokenized[i]

        return DataLoader(_DS(), batch_size=self.args.batch_size, shuffle=False,
                          collate_fn=collate_fn)

    # ── Hendrycks MATH ────────────────────────────────────────────────

    def _dl_math(self, max_samples, max_length, collate_fn):
        if hf_load_dataset is None:
            raise ImportError("datasets 라이브러리가 필요합니다")

        dataset_path = getattr(self.args, 'math_official_dataset_path', 'EleutherAI/hendrycks_math')
        subjects_arg = getattr(self.args, 'math_subjects', 'all')
        levels_arg   = getattr(self.args, 'math_levels',   'all')

        SUBJECTS = ['algebra', 'counting_and_probability', 'geometry',
                    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        target_subjects = SUBJECTS if subjects_arg == 'all' else \
            [s.strip().lower().replace(' ', '_') for s in subjects_arg.split(',')]
        target_levels   = None if levels_arg == 'all' else \
            {int(l.strip()) for l in levels_arg.split(',')}

        all_examples = []
        for subj in target_subjects:
            try:
                ds = hf_load_dataset(dataset_path, subj, split='train', trust_remote_code=True)
            except Exception as e:
                self.logger.warning(f"  MATH subject '{subj}' load failed: {e}")
                continue
            for ex in ds:
                if target_levels is not None:
                    try:
                        lvl = int(re.search(r'Level (\d+)', ex.get('level', '')).group(1))
                    except Exception:
                        continue
                    if lvl not in target_levels:
                        continue
                all_examples.append(ex)

        if max_samples > 0:
            all_examples = all_examples[:max_samples]

        def _extract_boxed(text: str) -> str:
            m = re.search(r'\\boxed\{(.+?)\}', text)
            return m.group(1) if m else text

        tokenized = []
        for ex in all_examples:
            problem  = ex.get('problem', ex.get('question', ''))
            solution = ex.get('solution', ex.get('answer', ''))
            answer   = _extract_boxed(solution)
            response = f"{solution}\n\nFinal Answer: ${answer}$"
            tokenized.append(self._tokenize_qa(problem, response, max_length))

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(tokenized)
            def __getitem__(self, i): return tokenized[i]

        return DataLoader(_DS(), batch_size=self.args.batch_size, shuffle=False,
                          collate_fn=collate_fn)

    # ── MetaMath ──────────────────────────────────────────────────────

    def _dl_metamath(self, max_samples, max_length, collate_fn):
        if hf_load_dataset is None:
            raise ImportError("datasets 라이브러리가 필요합니다")
        raw = hf_load_dataset('meta-math/MetaMathQA', split='train')
        if max_samples > 0:
            raw = raw.select(range(min(max_samples, len(raw))))

        tokenized = [self._tokenize_qa(ex['query'], ex['response'], max_length) for ex in raw]

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(tokenized)
            def __getitem__(self, i): return tokenized[i]

        return DataLoader(_DS(), batch_size=self.args.batch_size, shuffle=False,
                          collate_fn=collate_fn)

    # ── Wikipedia ─────────────────────────────────────────────────────

    def _dl_wikipedia(self, max_samples, max_length, collate_fn, seed):
        if hf_load_dataset is None:
            raise ImportError("datasets 라이브러리가 필요합니다")
        ds = hf_load_dataset('wikimedia/wikipedia', '20231101.en', split='train',
                             cache_dir=os.path.join(os.getcwd(), 'wikipedia_cache'))
        total = len(ds)
        n = min(max_samples, total) if max_samples > 0 else min(1000, total)
        random.seed(seed)
        idxs = random.sample(range(total), n)

        tokenized = []
        for idx in idxs:
            text = ds[idx]['text'].strip()
            if not text:
                continue
            encoding = self.tokenizer(text, max_length=max_length, truncation=True,
                                      padding=False, return_tensors='pt')
            tokenized.append({
                'input_ids':      encoding['input_ids'].squeeze(0).tolist(),
                'attention_mask': encoding['attention_mask'].squeeze(0).tolist(),
            })

        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(tokenized)
            def __getitem__(self, i): return tokenized[i]

        return DataLoader(_DS(), batch_size=self.args.batch_size, shuffle=False,
                          collate_fn=collate_fn,
                          generator=torch.Generator().manual_seed(seed))

    # ------------------------------------------------------------------
    # 4. Importance 계산
    # ------------------------------------------------------------------

    def _collect_warp_modules(self):
        """모델에서 WaRPModule 목록과 (layer_idx, layer_type) → module 매핑 반환."""
        target_indices = self._parse_target_layers(len(self.model.model.layers))
        mod_to_key = OrderedDict()
        for layer_idx in target_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in self.layer_types:
                mod = self._get_warp_module(layer, layer_type)
                if mod is not None:
                    mod_to_key[mod] = (layer_idx, layer_type)
        return mod_to_key

    def _compute_importance(self, dataloader: DataLoader, desc: str) -> Dict[Tuple, np.ndarray]:
        """
        model.eval() 모드에서 basis_coeff gradient를 누적하여 importance 계산.
        optimizer.step() 없음.
        """
        self.model.eval()

        mod_to_key = self._collect_warp_modules()
        warp_mods  = list(mod_to_key.keys())

        # 모든 파라미터 freeze → basis_coeff만 requires_grad
        for p in self.model.parameters():
            p.requires_grad_(False)
        for mod in warp_mods:
            mod.coeff_mask.data.zero_()
            if hasattr(mod, 'mask_mode'):
                mod.mask_mode.fill_(1)
            mod.basis_coeff.requires_grad_(True)

        acc: Dict = {}   # mod → acc importance (float32 tensor)
        total_samples = 0
        total_tokens  = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
            input_ids      = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels         = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.model.device)
            else:
                # labels 없으면 next-token prediction (패딩 위치 제외)
                labels = input_ids.masked_fill(attention_mask == 0, -100)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 labels=labels, use_cache=False)
            loss        = outputs.loss
            valid_toks  = (labels[:, 1:] != -100).sum().item()

            if valid_toks > 0 and loss is not None and torch.isfinite(loss):
                loss.backward()
                for mod in warp_mods:
                    if mod.basis_coeff.grad is not None:
                        g = mod.basis_coeff.grad.detach().abs().float()
                        acc[mod] = acc[mod].add_(g) if mod in acc else g.clone()
                total_samples += len(input_ids)
                total_tokens  += int(valid_toks)
                self.model.zero_grad(set_to_none=True)

            del outputs, loss, labels
            if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        self.logger.info(
            f"  ✓ {desc}: {total_samples} samples, {total_tokens:,} tokens"
        )

        # tensor → numpy, key 변환
        result = {}
        for mod, key in mod_to_key.items():
            if mod in acc:
                result[key] = acc[mod].cpu().numpy()

        # 재사용을 위해 basis_coeff requires_grad 유지 → 다음 패스 전에 다시 설정됨
        return result

    def compute_both_importances(self):
        """Preserve + Adapt 두 데이터셋의 importance를 순차적으로 계산."""
        preserve_ds   = getattr(self.args, 'preserve_dataset',  'circuit_breakers')
        adapt_ds      = getattr(self.args, 'adapt_dataset',     'gsm8k')
        preserve_samp = getattr(self.args, 'preserve_samples',  0)
        adapt_samp    = getattr(self.args, 'adapt_samples',     0)

        self.logger.info("=" * 70)
        self.logger.info(f"[PASS 1] Preserve dataset: {preserve_ds} (samples={preserve_samp or 'all'})")
        dl_preserve = self._build_dataloader(preserve_ds, preserve_samp)
        self.imp_preserve = self._compute_importance(dl_preserve, f"preserve({preserve_ds})")

        # 메모리 정리
        del dl_preserve
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("=" * 70)
        self.logger.info(f"[PASS 2] Adapt dataset: {adapt_ds} (samples={adapt_samp or 'all'})")
        dl_adapt = self._build_dataloader(adapt_ds, adapt_samp)
        self.imp_adapt = self._compute_importance(dl_adapt, f"adapt({adapt_ds})")

        del dl_adapt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("=" * 70)
        self.logger.info("✓ Both importance computations done")

    # ------------------------------------------------------------------
    # 5. Mask 생성
    # ------------------------------------------------------------------

    def generate_masks(self):
        keep_ratio = getattr(self.args, 'keep_ratio', 0.1)
        self.logger.info(f"Generating masks (keep_ratio={keep_ratio}) ...")
        for key, imp in self.imp_preserve.items():
            thr = np.quantile(imp, 1.0 - keep_ratio)
            self.mask_preserve[key] = (imp >= thr)
        for key, imp in self.imp_adapt.items():
            thr = np.quantile(imp, 1.0 - keep_ratio)
            self.mask_adapt[key] = (imp >= thr)
        self.logger.info(f"  preserve masks: {len(self.mask_preserve)}, "
                         f"adapt masks: {len(self.mask_adapt)}")

    # ------------------------------------------------------------------
    # 6. Overlap 분석
    # ------------------------------------------------------------------

    def analyze_overlap(self) -> dict:
        """
        레이어별 overlap 지표를 계산하고 dict로 반환.
        반환값: {key_str: {jaccard, adapt_blocked_ratio, preserve_also_adapt, ...}}
        """
        self.logger.info("=" * 70)
        self.logger.info("Overlap Analysis")
        self.logger.info("=" * 70)

        layer_stats = {}
        all_inter, all_p, all_a, all_union = 0, 0, 0, 0
        pearson_list = []

        common_keys = sorted(set(self.mask_preserve) & set(self.mask_adapt))
        self.logger.info(f"  Common (layer, type) keys: {len(common_keys)}")
        self.logger.info("")

        col_w = 10
        header = (f"{'Layer':>5}  {'Type':<9}  "
                  f"{'|P|':>{col_w}}  {'|A|':>{col_w}}  "
                  f"{'|P∩A|':>{col_w}}  {'Jaccard':>8}  "
                  f"{'A_blocked%':>10}  {'P→adapt%':>10}  "
                  f"{'Pearson_r':>10}")
        self.logger.info(header)
        self.logger.info("-" * len(header))

        for (layer_idx, layer_type) in common_keys:
            key   = (layer_idx, layer_type)
            P     = self.mask_preserve[key].astype(bool)
            A     = self.mask_adapt[key].astype(bool)
            inter = int((P & A).sum())
            union = int((P | A).sum())
            p_cnt = int(P.sum())
            a_cnt = int(A.sum())

            jaccard             = inter / union if union > 0 else 0.0
            adapt_blocked_ratio = inter / a_cnt if a_cnt > 0 else 0.0  # A 중 P에 막히는 비율
            preserve_also_adapt = inter / p_cnt if p_cnt > 0 else 0.0  # P 중 adapt에도 중요한 비율

            # Pearson correlation of raw scores
            imp_p = self.imp_preserve[key].flatten().astype(np.float64)
            imp_a = self.imp_adapt[key].flatten().astype(np.float64)
            corr  = float(np.corrcoef(imp_p, imp_a)[0, 1]) if imp_p.std() > 0 and imp_a.std() > 0 else 0.0
            pearson_list.append(corr)

            all_inter += inter
            all_p     += p_cnt
            all_a     += a_cnt
            all_union += union

            key_str = f"{layer_idx:02d}_{layer_type}"
            layer_stats[key_str] = {
                'layer_idx':            layer_idx,
                'layer_type':           layer_type,
                'preserve_count':       p_cnt,
                'adapt_count':          a_cnt,
                'intersection':         inter,
                'union':                union,
                'total_elements':       int(P.size),
                'jaccard':              round(jaccard,              6),
                'adapt_blocked_ratio':  round(adapt_blocked_ratio,  6),
                'preserve_also_adapt':  round(preserve_also_adapt,  6),
                'pearson_r':            round(corr,                 6),
            }

            self.logger.info(
                f"  {layer_idx:>3d}  {layer_type:<9s}  "
                f"  {p_cnt:>{col_w},d}  {a_cnt:>{col_w},d}  "
                f"  {inter:>{col_w},d}  {jaccard:>8.4f}  "
                f"  {adapt_blocked_ratio*100:>9.2f}%  "
                f"  {preserve_also_adapt*100:>9.2f}%  "
                f"  {corr:>10.4f}"
            )

        # 전체 집계
        global_jaccard = all_inter / all_union if all_union > 0 else 0.0
        global_adapt_blocked = all_inter / all_a if all_a > 0 else 0.0
        global_preserve_also_adapt = all_inter / all_p if all_p > 0 else 0.0
        global_pearson = float(np.mean(pearson_list)) if pearson_list else 0.0

        self.logger.info("-" * len(header))
        self.logger.info(
            f"  [GLOBAL] |P|={all_p:,}  |A|={all_a:,}  |P∩A|={all_inter:,}  "
            f"Jaccard={global_jaccard:.4f}  "
            f"A_blocked={global_adapt_blocked*100:.2f}%  "
            f"P→adapt={global_preserve_also_adapt*100:.2f}%  "
            f"mean_Pearson_r={global_pearson:.4f}"
        )
        self.logger.info("=" * 70)

        # 해석 출력
        self.logger.info("")
        self.logger.info("【해석】")
        self.logger.info(
            f"  - Jaccard: {global_jaccard:.4f} "
            f"({'높음 — 두 mask가 비슷한 파라미터를 중요하게 봄' if global_jaccard > 0.3 else '낮음 — 두 mask가 서로 다른 파라미터를 중요하게 봄'})"
        )
        self.logger.info(
            f"  - A_blocked: {global_adapt_blocked*100:.1f}% "
            f"— adapt에서 중요한 파라미터 중 {global_adapt_blocked*100:.1f}%가 preserve mask에 의해 freeze됨"
        )
        self.logger.info(
            f"  - P→adapt: {global_preserve_also_adapt*100:.1f}% "
            f"— preserve가 freeze한 파라미터 중 {global_preserve_also_adapt*100:.1f}%가 adapt에도 중요"
        )
        if global_adapt_blocked > 0.3:
            self.logger.info(
                "  ⚠️  A_blocked 30% 초과 → Phase 3 FT 시 WaRP freeze가 downstream 성능을 크게 방해할 가능성"
            )
        else:
            self.logger.info(
                "  ✅  A_blocked < 30% → preserve freeze가 downstream task에 미치는 방해 제한적"
            )

        global_stats = {
            'global_preserve_count':            all_p,
            'global_adapt_count':               all_a,
            'global_intersection':              all_inter,
            'global_union':                     all_union,
            'global_jaccard':                   round(global_jaccard,              6),
            'global_adapt_blocked_ratio':       round(global_adapt_blocked,        6),
            'global_preserve_also_adapt_ratio': round(global_preserve_also_adapt,  6),
            'global_mean_pearson_r':            round(global_pearson,              6),
        }

        return {'layers': layer_stats, 'global': global_stats}

    # ------------------------------------------------------------------
    # 7. 결과 저장
    # ------------------------------------------------------------------

    def save_results(self, stats: dict, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        keep_ratio     = getattr(self.args, 'keep_ratio',       0.1)
        preserve_ds    = getattr(self.args, 'preserve_dataset', 'circuit_breakers')
        adapt_ds       = getattr(self.args, 'adapt_dataset',    'gsm8k')
        preserve_samp  = getattr(self.args, 'preserve_samples', 0)
        adapt_samp     = getattr(self.args, 'adapt_samples',    0)

        # metadata
        metadata = {
            'phase0_model':      str(self.phase0_model_dir),
            'basis_dir':         str(self.basis_dir),
            'layer_types':       self.layer_types,
            'target_layers':     self.args.target_layers,
            'keep_ratio':        keep_ratio,
            'preserve_dataset':  preserve_ds,
            'adapt_dataset':     adapt_ds,
            'preserve_samples':  preserve_samp,
            'adapt_samples':     adapt_samp,
            'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # importance arrays (numpy로 저장)
        torch.save(
            {str(k): v for k, v in self.imp_preserve.items()},
            os.path.join(out_dir, 'importances_preserve.pt')
        )
        torch.save(
            {str(k): v for k, v in self.imp_adapt.items()},
            os.path.join(out_dir, 'importances_adapt.pt')
        )

        # overlap stats JSON
        with open(os.path.join(out_dir, 'overlap_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        # CSV
        import csv
        csv_path = os.path.join(out_dir, 'overlap_stats.csv')
        layer_rows = list(stats['layers'].values())
        if layer_rows:
            fieldnames = list(layer_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(layer_rows)

        # mask 저장 (preserve)
        self._save_masks(self.mask_preserve, os.path.join(out_dir, 'masks_preserve'),
                         dataset=preserve_ds)
        self._save_masks(self.mask_adapt,    os.path.join(out_dir, 'masks_adapt'),
                         dataset=adapt_ds)

        self.logger.info(f"✓ Results saved to: {out_dir}")
        self.logger.info(f"  - metadata.json")
        self.logger.info(f"  - importances_preserve.pt / importances_adapt.pt")
        self.logger.info(f"  - overlap_stats.json / .csv")
        self.logger.info(f"  - masks_preserve/ masks_adapt/")

    def _save_masks(self, masks: dict, masks_dir: str, dataset: str):
        os.makedirs(masks_dir, exist_ok=True)
        for (layer_idx, layer_type), mask in masks.items():
            lt_dir = os.path.join(masks_dir, layer_type)
            os.makedirs(lt_dir, exist_ok=True)
            torch.save({'mask': mask.astype(np.float32)},
                       os.path.join(lt_dir, f'layer_{layer_idx:02d}_mask.pt'))
        meta = {
            'keep_ratio':       getattr(self.args, 'keep_ratio', 0.1),
            'masking_strategy': 'per_layer',
            'layer_types':      self.layer_types,
            'dataset':          dataset,
        }
        with open(os.path.join(masks_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # 8. 시각화
    # ------------------------------------------------------------------

    def plot_results(self, stats: dict, out_dir: str):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available — skipping plots")
            return

        figures_dir = os.path.join(out_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        preserve_ds = getattr(self.args, 'preserve_dataset', 'circuit_breakers')
        adapt_ds    = getattr(self.args, 'adapt_dataset',    'gsm8k')

        layer_data = list(stats['layers'].values())
        # 레이어 × layer_type 조합으로 정렬
        layer_data.sort(key=lambda x: (x['layer_idx'], x['layer_type']))

        labels     = [f"{d['layer_idx']:02d}\n{d['layer_type'][:6]}" for d in layer_data]
        jaccard    = [d['jaccard']             for d in layer_data]
        a_blocked  = [d['adapt_blocked_ratio'] for d in layer_data]
        p_to_adapt = [d['preserve_also_adapt'] for d in layer_data]
        pearson    = [d['pearson_r']           for d in layer_data]

        x = np.arange(len(labels))

        # ── Figure 1: 레이어별 overlap 막대 그래프 ──────────────────────
        fig, axes = plt.subplots(4, 1, figsize=(max(16, len(labels) * 0.6), 20))
        fig.suptitle(
            f"Dual Importance Overlap\nPreserve: {preserve_ds}  |  Adapt: {adapt_ds}  "
            f"|  keep_ratio={getattr(self.args, 'keep_ratio', 0.1):.2f}",
            fontsize=13
        )

        def _bar(ax, values, title, ylabel, color, ylim=(0, 1)):
            ax.bar(x, values, color=color, alpha=0.75, edgecolor='black', linewidth=0.4)
            g_val = stats['global'].get(
                f"global_{title.lower().replace(' ', '_')}", None)
            if g_val is not None:
                ax.axhline(g_val, color='red', linestyle='--', linewidth=1.2,
                           label=f"global={g_val:.4f}")
                ax.legend(fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylim(*ylim)
            ax.grid(axis='y', alpha=0.3)

        _bar(axes[0], jaccard,    'Jaccard (|P∩A|/|P∪A|)',
             'Jaccard', 'steelblue')
        _bar(axes[1], a_blocked,  'A_blocked_ratio (|P∩A|/|A|)',
             'Ratio', 'tomato')
        _bar(axes[2], p_to_adapt, 'Preserve→Adapt ratio (|P∩A|/|P|)',
             'Ratio', 'seagreen')
        p_ylim = (min(min(pearson) - 0.05, -0.1), max(max(pearson) + 0.05, 0.5))
        _bar(axes[3], pearson,    'Pearson_r (importance score correlation)',
             'r', 'mediumpurple', ylim=p_ylim)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(figures_dir, 'overlap_by_layer.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"  saved: {path}")

        # ── Figure 2: 전체 scatter (log-scale) ────────────────────────
        all_p_vals, all_a_vals, all_colors = [], [], []
        for key in sorted(set(self.imp_preserve) & set(self.imp_adapt)):
            imp_p = self.imp_preserve[key].flatten().astype(np.float64)
            imp_a = self.imp_adapt[key].flatten().astype(np.float64)
            mp    = self.mask_preserve[key].flatten().astype(bool)
            ma    = self.mask_adapt[key].flatten().astype(bool)
            cats  = np.zeros(len(imp_p), dtype=int)
            cats[mp  & ~ma] = 1   # preserve only
            cats[~mp &  ma] = 2   # adapt only
            cats[mp  &  ma] = 3   # both
            all_p_vals.append(imp_p)
            all_a_vals.append(imp_a)
            all_colors.append(cats)

        if all_p_vals:
            p_all = np.concatenate(all_p_vals)
            a_all = np.concatenate(all_a_vals)
            c_all = np.concatenate(all_colors)

            # 샘플링 (너무 많으면 느림)
            MAX_SCATTER = 200_000
            if len(p_all) > MAX_SCATTER:
                rng  = np.random.default_rng(0)
                idxs = rng.choice(len(p_all), MAX_SCATTER, replace=False)
                p_all, a_all, c_all = p_all[idxs], a_all[idxs], c_all[idxs]

            eps = 1e-12
            fig, ax = plt.subplots(figsize=(8, 7))
            cmap   = {0: ('lightgray', 'neither', 0.15),
                      1: ('steelblue', 'preserve only', 0.5),
                      2: ('tomato',    'adapt only',    0.5),
                      3: ('purple',    'both',          0.8)}
            for cat_id, (color, label, alpha) in cmap.items():
                mask = c_all == cat_id
                if mask.sum() == 0:
                    continue
                ax.scatter(np.log10(p_all[mask] + eps), np.log10(a_all[mask] + eps),
                           s=1, alpha=alpha, color=color, label=label, rasterized=True)
            ax.set_xlabel(f"log10(importance_preserve)  [{preserve_ds}]", fontsize=11)
            ax.set_ylabel(f"log10(importance_adapt)  [{adapt_ds}]",       fontsize=11)
            ax.set_title("Global importance scatter (all layers)\n"
                         f"keep_ratio={getattr(self.args,'keep_ratio',0.1):.2f}", fontsize=11)
            ax.legend(markerscale=6, fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(figures_dir, 'scatter_global.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"  saved: {path}")

        # ── Figure 3: Jaccard 히트맵 (layer_idx × layer_type) ─────────
        unique_layers = sorted({d['layer_idx'] for d in layer_data})
        unique_types  = sorted({d['layer_type'] for d in layer_data})
        heat = np.full((len(unique_layers), len(unique_types)), np.nan)
        for d in layer_data:
            ri = unique_layers.index(d['layer_idx'])
            ci = unique_types.index(d['layer_type'])
            heat[ri, ci] = d['jaccard']

        if not np.all(np.isnan(heat)):
            fig, ax = plt.subplots(figsize=(max(6, len(unique_types) * 1.2),
                                            max(6, len(unique_layers) * 0.4)))
            im = ax.imshow(heat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
            ax.set_xticks(range(len(unique_types)))
            ax.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(unique_layers)))
            ax.set_yticklabels([str(l) for l in unique_layers], fontsize=8)
            ax.set_title(
                f"Jaccard heatmap\nPreserve: {preserve_ds}  |  Adapt: {adapt_ds}", fontsize=11
            )
            ax.set_xlabel('Layer Type', fontsize=10)
            ax.set_ylabel('Layer Index', fontsize=10)

            # 셀 값 텍스트
            for ri in range(len(unique_layers)):
                for ci in range(len(unique_types)):
                    if not np.isnan(heat[ri, ci]):
                        ax.text(ci, ri, f"{heat[ri, ci]:.3f}", ha='center', va='center',
                                fontsize=7 if len(unique_layers) < 20 else 5,
                                color='black' if heat[ri, ci] < 0.4 else 'white')

            plt.colorbar(im, ax=ax, label='Jaccard', fraction=0.02, pad=0.04)
            plt.tight_layout()
            path = os.path.join(figures_dir, 'heatmap_jaccard.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"  saved: {path}")

        # ── Figure 4: A_blocked 히트맵 ────────────────────────────────
        heat2 = np.full_like(heat, np.nan)
        for d in layer_data:
            ri = unique_layers.index(d['layer_idx'])
            ci = unique_types.index(d['layer_type'])
            heat2[ri, ci] = d['adapt_blocked_ratio']

        if not np.all(np.isnan(heat2)):
            fig, ax = plt.subplots(figsize=(max(6, len(unique_types) * 1.2),
                                            max(6, len(unique_layers) * 0.4)))
            im = ax.imshow(heat2, aspect='auto', cmap='Reds', vmin=0, vmax=0.5)
            ax.set_xticks(range(len(unique_types)))
            ax.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(unique_layers)))
            ax.set_yticklabels([str(l) for l in unique_layers], fontsize=8)
            ax.set_title(
                f"Adapt_blocked_ratio heatmap  (|P∩A|/|A|)\n"
                f"Preserve: {preserve_ds}  |  Adapt: {adapt_ds}", fontsize=11
            )
            ax.set_xlabel('Layer Type', fontsize=10)
            ax.set_ylabel('Layer Index', fontsize=10)
            for ri in range(len(unique_layers)):
                for ci in range(len(unique_types)):
                    if not np.isnan(heat2[ri, ci]):
                        ax.text(ci, ri, f"{heat2[ri,ci]:.3f}", ha='center', va='center',
                                fontsize=7 if len(unique_layers) < 20 else 5,
                                color='black' if heat2[ri, ci] < 0.35 else 'white')
            plt.colorbar(im, ax=ax, label='A_blocked', fraction=0.02, pad=0.04)
            plt.tight_layout()
            path = os.path.join(figures_dir, 'heatmap_adapt_blocked.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"  saved: {path}")

        self.logger.info(f"  ✓ Figures saved to: {figures_dir}")

    # ------------------------------------------------------------------
    # 9. 전체 파이프라인
    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        전체 파이프라인 실행.
        반환값: out_dir (결과 저장 경로)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = os.path.join(
            getattr(self.args, 'output_dir', './analysis'),
            f"dual_importance_{timestamp}"
        )

        self.logger.info("=" * 70)
        self.logger.info("Dual Importance Analysis (WaRP Basis-Rotated Space)")
        self.logger.info("=" * 70)

        self.load_basis()
        self.load_model()
        self.compute_both_importances()
        self.generate_masks()
        stats = self.analyze_overlap()

        self.logger.info(f"Saving results to: {out_dir}")
        self.save_results(stats, out_dir)

        if not getattr(self.args, 'no_plot', False):
            self.logger.info("Generating plots ...")
            self.plot_results(stats, out_dir)

        self.logger.info("=" * 70)
        self.logger.info(f"✓ Dual Importance Analysis DONE")
        self.logger.info(f"  Output: {out_dir}")
        self.logger.info("=" * 70)
        return out_dir

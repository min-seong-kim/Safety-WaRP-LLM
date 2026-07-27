"""
Phase 1 (variant): ActSVD **출력측** 기저 / WSR **입력측** 기저를 한 코드로 구성.

`wsr_actsvd_ablation_spec.md` §3 Arm B, §8 항목 5 구현.

────────────────────────────────────────────────────────────────────────────
두 기저의 차이 (절대 섞지 말 것)

  W ∈ R^{m×n},  X_in ∈ R^{n×M} = safety corpus의 layer 입력 활성화

  · 입력측 (WSR-Tune, 기존 Phase 1):
        X_in X_in^T = U_in Λ U_in^T,      U_in ∈ R^{n×n}
        W̃ = W U_in                        (RIGHT multiply)

  · 출력측 (ActSVD, Wei et al. 2024 §2.1):
        U S V^T ≈ W X_in,                 U_out ∈ R^{m×m} = left singular vectors
        Ŵ = U_out[:, :r] U_out[:, :r]^T W (LEFT multiply / projection)
        W̃ = U_out^T W                     ← 본 저장소에서 rank freezing을 위해 쓰는 좌표계

  구현 요령: Y = W X_in 의 left singular vector = Y Y^T 의 eigenvector 이므로,
  출력 활성화 Y의 Gram Y Y^T ∈ R^{m×m}를 누적한 뒤 대칭행렬 SVD 하면 U_out을 얻는다.
  (LLaMA의 q/k/v/up/down projection은 bias가 없어 hook의 output이 곧 W x 이다.
   bias가 있으면 ActSVD 정의와 어긋나므로 예외를 던진다.)

Wei et al. footnote 9은 "freezing 연산은 U^u, U^s로는 쉽게 달성할 수 없다"고 적었다.
좌표계를 U_out^T W 로 옮겨 두면 rank 동결이 단순한 **행 마스킹**이 된다 — 그 재파라미터화가
바로 이 파일이 만드는 산출물이다.

────────────────────────────────────────────────────────────────────────────
기존 Phase 1과 의도적으로 다른 점

  1. Gram 누적을 fp32로 한다 (기존은 모델 dtype=bf16 누적). 출력측 활성화는
     스케일이 커서 bf16 누적의 오차가 크다. `--gram_dtype bfloat16`으로 레거시 동작 복원 가능.
  2. `--basis_token_scope response` 로 응답 토큰만 사용할 수 있다 (spec §2).
     기본값은 `all` — 논문 Table 2/5 재현(정합성 체크 §8 항목 1·2)을 깨지 않기 위해서다.
  3. 저장 dtype을 고를 수 있다 (기본 bfloat16, spec §6). Phase 2/3은 어차피 모델 dtype으로
     캐스팅해 쓰므로 bf16 저장이 정보 손실을 추가하지 않는다.

산출물 레이아웃은 기존 Phase 1과 동일하므로 Phase 2/3이 그대로 읽는다:
  <out>/basis/<layer_type>/layer_NN_svd.pt   {'U','S','UT'}
  <out>/basis/metadata.json                  (+ basis_side / token_scope / diagnostics)
"""

import json
import os
from datetime import datetime

import torch
from tqdm import tqdm

from .phase1_basis import Phase1BasisBuilder

_DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


class ActSVDBasisBuilder(Phase1BasisBuilder):
    """입력측(WSR) / 출력측(ActSVD) 활성화 기저 빌더."""

    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.basis_side = getattr(args, 'basis_side', 'input') or 'input'
        if self.basis_side not in ('input', 'output'):
            raise ValueError(
                f"--basis_side must be 'input' or 'output', got {self.basis_side!r}"
            )

        self.token_scope = getattr(args, 'basis_token_scope', 'all') or 'all'
        if self.token_scope not in ('all', 'response'):
            raise ValueError(
                f"--basis_token_scope must be 'all' or 'response', got {self.token_scope!r}"
            )

        self.gram_dtype = _DTYPE_MAP[getattr(args, 'gram_dtype', 'float32') or 'float32']
        self.save_dtype = _DTYPE_MAP[getattr(args, 'basis_save_dtype', 'bfloat16') or 'bfloat16']

        # 진단 정보 (metadata.json에 기록)
        self.diagnostics = {}

        # checkpoint 디렉토리 이름에 side를 박아 두어 입력/출력 기저가 섞이지 않게 한다.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_base = os.path.join(
            getattr(args, 'output_dir', './checkpoints'),
            f'phase1_{self.basis_side}_{timestamp}',
        )
        os.makedirs(checkpoint_base, exist_ok=True)
        self.checkpoint_dir = os.path.join(checkpoint_base, 'basis')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger.info("=" * 70)
        self.logger.info(f"ActSVD/WSR basis builder — side={self.basis_side}")
        self.logger.info(f"  - token scope   : {self.token_scope}")
        self.logger.info(f"  - gram dtype    : {self.gram_dtype}")
        self.logger.info(f"  - save dtype    : {self.save_dtype}")
        self.logger.info(f"  - output dir    : {self.checkpoint_dir}")
        self.logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # 데이터: 응답 토큰만 쓰는 경우를 위해 response_mask를 함께 만든다
    # ──────────────────────────────────────────────────────────────────

    def _load_circuit_breakers(self):
        if self.token_scope == 'all':
            # 기존 Phase 1과 완전히 동일한 경로 (재현성 유지)
            return super()._load_circuit_breakers()

        path = getattr(self.args, 'circuit_breakers_path', './data/circuit_breakers_train.json')
        self.logger.info(f"Loading circuit_breakers (response-only scope) from {path}...")

        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        max_samples = getattr(self.args, 'circuit_breakers_samples_phase1', 4994)
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]
        self.logger.info(f"✓ Dataset loaded: {len(dataset)} samples")

        builder = self
        max_length = getattr(self.args, 'max_length', 1024)

        class ResponseScopedDataset(torch.utils.data.Dataset):
            """prompt 길이를 따로 토큰화해 response 구간만 1인 마스크를 만든다."""

            def __init__(self, items, tokenizer):
                self.items = items
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                item = self.items[idx]
                prompt = item.get('prompt', '')
                response = item.get('llama3_output', '')

                full_text = builder._format_phase1_text(prompt, response)
                prompt_text = builder._format_phase1_text(prompt, None)

                enc = self.tokenizer(full_text, truncation=True, max_length=max_length,
                                     return_tensors='pt')
                prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=max_length,
                                            return_tensors='pt')

                input_ids = enc['input_ids'].squeeze(0)
                attention_mask = enc['attention_mask'].squeeze(0)
                prompt_len = min(prompt_enc['input_ids'].size(1), input_ids.size(0))

                response_mask = attention_mask.clone()
                response_mask[:prompt_len] = 0

                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'response_mask': response_mask,
                }

        def collate_fn(batch):
            max_len = max(len(b['input_ids']) for b in batch)
            pad_id = self.tokenizer.pad_token_id
            out = {'input_ids': [], 'attention_mask': [], 'response_mask': []}
            for b in batch:
                pad = max_len - len(b['input_ids'])
                out['input_ids'].append(
                    torch.cat([b['input_ids'], torch.full((pad,), pad_id, dtype=b['input_ids'].dtype)])
                    if pad > 0 else b['input_ids'])
                for key in ('attention_mask', 'response_mask'):
                    t = b[key]
                    out[key].append(
                        torch.cat([t, torch.zeros(pad, dtype=t.dtype)]) if pad > 0 else t)
            return {k: torch.stack(v) for k, v in out.items()}

        wrapped = ResponseScopedDataset(dataset, self.tokenizer)
        self.dataloader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(112),
        )
        self.logger.info(f"✓ Dataloader created ({len(self.dataloader)} batches, response-only masking)")

        # prompt/response 경계가 실제로 맞는지 첫 샘플로 한 번 확인해 둔다.
        # (prompt-only 렌더가 full 렌더의 접두어라는 가정에 의존한다. Phase 2는
        #  add_generation_prompt=True를 쓰므로 경계 규약이 몇 토큰 다를 수 있다.)
        probe = wrapped[0]
        n_resp = int(probe['response_mask'].sum().item())
        n_all = int(probe['attention_mask'].sum().item())
        boundary = int((probe['response_mask'] == 0).sum().item())
        self.logger.info(f"  - sample 0: {n_resp}/{n_all} tokens scored (prompt prefix = {boundary} tokens)")
        self.logger.info(
            "  - response head: "
            + repr(self.tokenizer.decode(probe['input_ids'][boundary:boundary + 16]))
        )
        if n_resp == 0 or n_resp == n_all:
            raise ValueError(
                "response-only scope인데 응답 구간이 비었거나 전체와 같습니다. "
                "chat template 렌더링 규약을 확인하세요."
            )

    # ──────────────────────────────────────────────────────────────────
    # Gram 누적 (입력측 또는 출력측)
    # ──────────────────────────────────────────────────────────────────

    def collect_activations_and_accumulate_gram(self):
        side = self.basis_side
        self.logger.info("=" * 70)
        self.logger.info(f"Accumulating {side}-side activation Gram matrices")
        if side == 'output':
            self.logger.info("  Y = W X_in (hook output) → Gram = Y Y^T ∈ R^{m×m}")
            self.logger.info("  left singular vectors of Y == eigenvectors of Y Y^T  (= ActSVD U)")
        else:
            self.logger.info("  Gram = X_in X_in^T ∈ R^{n×n}  (= WSR-Tune U)")
        self.logger.info("=" * 70)

        num_layers = len(self.model.model.layers)
        layer_indices = self._parse_target_layers(num_layers)
        layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]

        for layer_idx in layer_indices:
            for layer_type in layer_types:
                self.gram_matrices[(layer_idx, layer_type)] = None
                self.num_samples[(layer_idx, layer_type)] = 0

        # 예상 메모리 사전 보고 (spec §6: peak GPU mem / basis storage를 반드시 로깅)
        est_bytes = 0
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in layer_types:
                mod = self._get_target_module_for_basis(layer, layer_type)
                dim = mod.out_features if side == 'output' else mod.in_features
                est_bytes += dim * dim * self.gram_dtype.itemsize
        self.logger.info(f"  - Estimated Gram memory: {est_bytes / 2**30:.2f} GiB "
                         f"({len(layer_indices) * len(layer_types)} modules, dtype={self.gram_dtype})")

        current_mask = {'val': None}
        hooks = []

        def make_hook(layer_idx, layer_type):
            def hook(module, inputs, output):
                act = output if side == 'output' else inputs[0]
                if not torch.is_tensor(act):
                    act = act[0]
                b, s, d = act.shape

                mask = current_mask['val']
                if mask is not None:
                    act = act * mask.unsqueeze(-1).to(device=act.device, dtype=act.dtype)

                flat = act.reshape(b * s, d).to(self.gram_dtype)
                gram = flat.t() @ flat

                key = (layer_idx, layer_type)
                if self.gram_matrices[key] is None:
                    self.gram_matrices[key] = gram
                else:
                    self.gram_matrices[key] += gram

                valid = int(mask.sum().item()) if mask is not None else b * s
                self.num_samples[key] += valid
            return hook

        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            for layer_type in layer_types:
                module = self._get_target_module_for_basis(layer, layer_type)
                if side == 'output' and getattr(module, 'bias', None) is not None:
                    raise ValueError(
                        f"layer {layer_idx} {layer_type}: bias가 있는 모듈은 출력측 기저를 "
                        "정의대로 계산할 수 없습니다 (hook output = Wx + b ≠ Wx). "
                        "ActSVD는 W X_in 의 SVD로 정의됩니다."
                    )
                hooks.append(module.register_forward_hook(make_hook(layer_idx, layer_type)))

        self.logger.info(f"✓ {len(hooks)} accumulation hooks registered")

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Gram ({side})", disable=not self.args.debug):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                token_mask = batch.get('response_mask', attention_mask).to(self.model.device)

                current_mask['val'] = token_mask
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                self.stats['total_samples'] += int(input_ids.shape[0])
                self.stats['total_tokens'] += int(token_mask.sum().item())

        for h in hooks:
            h.remove()

        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        self.diagnostics['peak_gpu_mem_gib_gram'] = round(peak, 3)

        self.logger.info("✓ Gram accumulation completed")
        self.logger.info(f"  - Total samples : {self.stats['total_samples']}")
        self.logger.info(f"  - Scored tokens : {self.stats['total_tokens']} (scope={self.token_scope})")
        self.logger.info(f"  - Peak GPU mem  : {peak:.2f} GiB")

    # ──────────────────────────────────────────────────────────────────
    # SVD + 저장
    # ──────────────────────────────────────────────────────────────────

    def compute_svd(self):
        self.logger.info("=" * 70)
        self.logger.info(f"Computing SVD of {self.basis_side}-side Gram matrices")
        self.logger.info("=" * 70)

        per_layer_diag = {}
        layers_saved = 0

        for layer_idx, layer_type in sorted(self.gram_matrices.keys()):
            gram = self.gram_matrices[(layer_idx, layer_type)]
            if gram is None:
                self.logger.warning(f"Layer {layer_idx} ({layer_type}): Gram 없음, 건너뜀")
                continue

            gram_f = gram.float()
            U, S, UT = torch.linalg.svd(gram_f, full_matrices=False)
            # 대칭 PSD 행렬이므로 U == V. 기존 Phase 1과 동일하게 V(=UT^T)를 저장한다.
            V = UT.t()

            sym_err = (U - V).abs().max().item()

            # 저장 dtype으로 캐스팅한 뒤의 직교성 (Phase 2/3이 실제로 쓰는 정밀도)
            V_saved = V.to(self.save_dtype)
            probe = V_saved[:, :min(256, V_saved.shape[1])].float()
            ortho_err = (probe.t() @ probe - torch.eye(probe.shape[1], device=probe.device)).abs().max().item()

            total = float(S.sum().item())
            cum = torch.cumsum(S, dim=0) / max(total, 1e-30)
            # ActSVD 관점의 진단: Gram 고유값 λ = σ(WX)² 이므로
            # rank-r 절단의 출력 잔차 에너지 비율 = Σ_{i>r} λ_i / Σ_i λ_i
            energy = {}
            for r_frac in (0.01, 0.05, 0.10, 0.20):
                r = max(1, int(round(r_frac * len(S))))
                energy[f'top{int(r_frac * 100)}pct_energy'] = round(float(cum[r - 1].item()), 6)

            per_layer_diag[f'{layer_type}/layer_{layer_idx:02d}'] = {
                'dim': int(V.shape[0]),
                'symmetry_err': sym_err,
                'orthogonality_err_after_cast': ortho_err,
                **energy,
            }

            # 'UT' 는 저장하지 않는다. Phase 2/3은 'U'만 읽으며, V^T 는 U.t() 로 즉시 얻는다.
            # (레거시 Phase 1은 U와 UT를 각각 CPU로 복사해 저장해서 디스크를 정확히 2배 쓴다:
            #  Llama-2-7B 기준 11.2 GiB → 22.45 GiB. 여기서는 그 낭비를 없앤다.)
            self._save_svd_result(layer_idx, layer_type, {
                'U': V_saved.cpu(),
                'S': S.to(torch.float32).cpu(),
                'UT': None,
            })
            layers_saved += 1

            self.logger.info(
                f"Layer {layer_idx:2d} ({layer_type:9s}) dim={V.shape[0]:5d} "
                f"| sym={sym_err:.2e} ortho={ortho_err:.2e} "
                f"| energy@10%={energy['top10pct_energy']:.4f}"
            )

            del gram_f, U, S, UT, V, V_saved
            self.gram_matrices[(layer_idx, layer_type)] = None

        self.gram_matrices.clear()
        self.num_samples.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.diagnostics['per_layer'] = per_layer_diag
        self.diagnostics['num_layers_saved'] = layers_saved
        self.logger.info(f"✓ SVD completed: {layers_saved} modules saved")

    def _save_svd_result(self, layer_idx, layer_type, svd_result):
        layer_type_dir = os.path.join(self.checkpoint_dir, layer_type)
        os.makedirs(layer_type_dir, exist_ok=True)
        save_path = os.path.join(layer_type_dir, f'layer_{layer_idx:02d}_svd.pt')
        torch.save({
            'U': svd_result['U'],
            'S': svd_result['S'],
            'UT': svd_result['UT'],
            'basis_side': self.basis_side,
        }, save_path)

    def save_basis(self):
        """부모의 검증 로깅을 그대로 쓰고, metadata에 ablation 필드를 덧붙인다."""
        basis_dir = super().save_basis()

        metadata_path = os.path.join(basis_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        storage_bytes = 0
        for root, _dirs, files in os.walk(basis_dir):
            for name in files:
                if name.endswith('.pt'):
                    storage_bytes += os.path.getsize(os.path.join(root, name))

        metadata.update({
            'basis_side': self.basis_side,
            'basis_space': 'output (ActSVD left singular vectors of W X_in)'
                           if self.basis_side == 'output'
                           else 'input (eigenbasis of X_in X_in^T)',
            'token_scope': self.token_scope,
            'gram_dtype': str(self.gram_dtype),
            'save_dtype': str(self.save_dtype),
            'safety_dataset': getattr(self.args, 'safety_dataset', 'circuit_breakers'),
            'storage_bytes': storage_bytes,
            'storage_gib': round(storage_bytes / 2**30, 3),
            'diagnostics': self.diagnostics,
            'builder': 'ActSVDBasisBuilder',
        })
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✓ basis metadata updated (side={self.basis_side}, "
                         f"storage={metadata['storage_gib']:.2f} GiB)")
        return basis_dir

    # ──────────────────────────────────────────────────────────────────

    def _get_target_module_for_basis(self, layer, layer_type):
        if layer_type == 'ffn_down':
            return layer.mlp.down_proj
        if layer_type == 'ffn_gate':
            return layer.mlp.gate_proj
        if layer_type == 'ffn_up':
            return layer.mlp.up_proj
        if layer_type == 'attn_q':
            return layer.self_attn.q_proj
        if layer_type == 'attn_k':
            return layer.self_attn.k_proj
        if layer_type == 'attn_v':
            return layer.self_attn.v_proj
        if layer_type == 'attn_o':
            return layer.self_attn.o_proj
        raise ValueError(f"Unknown layer type: {layer_type}")

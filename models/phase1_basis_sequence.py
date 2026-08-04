"""
Phase 1 (Sequence-wise variant): Basis Construction

기본 Phase1BasisBuilder는 활성화를 **token-wise**로 수집한다
(모든 non-pad 토큰 위치 하나하나가 공분산의 샘플 φ → Gram = Σ_토큰 φφᵀ).

본 변형은 각 시퀀스를 **하나의 벡터로 pooling**한 뒤 공분산을 누적한다
(원논문 WaRP의 per-sample 공식 Φ=[φ(x_1),…,φ(x_N)]에 더 가까움).

    Gram = Σ_시퀀스  φ̄(x) φ̄(x)ᵀ         (φ̄ = 시퀀스 pooling 벡터)

pooling 방식 (--seq_pool):
    mean : 유효 토큰 평균  φ̄ = (Σ_t m_t h_t) / (Σ_t m_t)     [기본]
    last : 마지막 유효 토큰 hidden (EOS 요약 표현, padding-agnostic)
    sum  : 유효 토큰 합   φ̄ = Σ_t m_t h_t

Gram 행렬 shape 은 (hidden, hidden)으로 token-wise 와 동일하므로
compute_svd / save_basis / Phase 2/3 는 **수정 없이 그대로 재사용**된다.
"""

import gc
import torch
from tqdm import tqdm

from .phase1_basis import Phase1BasisBuilder


class Phase1BasisBuilderSequence(Phase1BasisBuilder):
    """시퀀스 단위 pooling 으로 Gram 을 누적하는 Phase 1 빌더."""

    def _seq_pool_activations(self, act, mask, pool):
        """
        act : (batch, seq, hidden)   레이어 입력 활성화
        mask: (batch, seq) or None   attention_mask (1=유효 토큰, 0=pad)
        pool: 'mean' | 'last' | 'sum'
        반환: (batch, hidden)  시퀀스별 pooling 벡터
        """
        batch_size, seq_len, hidden_dim = act.shape

        if mask is None:
            # pad 정보가 없으면 전체 토큰 대상
            mask = torch.ones(batch_size, seq_len, device=act.device, dtype=act.dtype)
        else:
            mask = mask.to(device=act.device, dtype=act.dtype)

        if pool == 'last':
            # 마지막 유효 토큰 위치(오른쪽/왼쪽 padding 모두 대응):
            # 뒤집어서 첫 1의 위치 → 원본에서 마지막 1의 위치
            flipped = mask.flip(1)
            last_from_right = torch.argmax(flipped, dim=1)          # (batch,)
            last_pos = (seq_len - 1) - last_from_right              # (batch,)
            idx = torch.arange(batch_size, device=act.device)
            pooled = act[idx, last_pos]                             # (batch, hidden)
        else:
            m3 = mask.unsqueeze(-1)                                 # (batch, seq, 1)
            summed = (act * m3).sum(dim=1)                          # (batch, hidden)
            if pool == 'sum':
                pooled = summed
            else:  # mean (기본)
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1)
                pooled = summed / denom
        return pooled

    def collect_activations_and_accumulate_gram(self):
        """
        시퀀스 단위 Gram 누적.

        기본 빌더와 동일한 hook/loop 구조지만, 각 배치에서
        활성화를 시퀀스별로 pooling 한 뒤 Gram 에 누적한다.
        """
        try:
            pool = getattr(self.args, 'seq_pool', 'mean')
            self.logger.info("Collecting activations and accumulating Gram matrices (GPU)...")
            self.logger.info(f"✅ Sequence-wise pooling mode: pool='{pool}' "
                             f"(각 시퀀스 = 공분산 샘플 1개)")

            num_layers = len(self.model.model.layers)
            layer_indices = self._parse_target_layers(num_layers)
            layer_types = [lt.strip() for lt in self.args.layer_type.split(',')]

            self.logger.info(f"Target layers: {layer_indices}")
            self.logger.info(f"Layer types: {layer_types}")

            for layer_idx in layer_indices:
                for layer_type in layer_types:
                    self.gram_matrices[(layer_idx, layer_type)] = None
                    self.num_samples[(layer_idx, layer_type)] = 0

            current_mask = {'val': None}
            hooks = []

            def get_accumulation_hook(layer_idx, layer_type):
                def hook(module, input, output):
                    # input[0]: (batch, seq, hidden)
                    act = input[0]
                    mask = current_mask['val']  # (batch, seq)

                    # 시퀀스별 pooling → (batch, hidden)
                    pooled = self._seq_pool_activations(act, mask, pool)

                    # Gram 누적: Σ φ̄ φ̄ᵀ
                    gram_batch = pooled.t() @ pooled  # (hidden, hidden)

                    key = (layer_idx, layer_type)
                    if self.gram_matrices[key] is None:
                        self.gram_matrices[key] = gram_batch
                    else:
                        self.gram_matrices[key] += gram_batch

                    # 시퀀스(=샘플) 수 카운트
                    self.num_samples[key] += pooled.shape[0]
                return hook

            for layer_idx in layer_indices:
                layer = self.model.model.layers[layer_idx]
                for layer_type in layer_types:
                    if layer_type == 'ffn_down':
                        target_module = layer.mlp.down_proj
                    elif layer_type == 'ffn_gate':
                        target_module = layer.mlp.gate_proj
                    elif layer_type == 'ffn_up':
                        target_module = layer.mlp.up_proj
                    elif layer_type == 'attn_q':
                        target_module = layer.self_attn.q_proj
                    elif layer_type == 'attn_k':
                        target_module = layer.self_attn.k_proj
                    elif layer_type == 'attn_v':
                        target_module = layer.self_attn.v_proj
                    elif layer_type == 'attn_o':
                        target_module = layer.self_attn.o_proj
                    else:
                        raise ValueError(f"Unknown layer type: {layer_type}")

                    hook_handle = target_module.register_forward_hook(
                        get_accumulation_hook(layer_idx, layer_type)
                    )
                    hooks.append(hook_handle)

            self.logger.info(f"✓ {len(hooks)} accumulation hooks registered (sequence-wise)")

            with torch.no_grad():
                progress_bar = tqdm(
                    self.dataloader,
                    desc=f"Accumulating Gram (seq/{pool})",
                    disable=not self.args.debug
                )
                total_batches = 0
                for batch_idx, batch in enumerate(progress_bar):
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    current_mask['val'] = attention_mask

                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    total_batches += 1
                    self.stats['total_samples'] += input_ids.shape[0]
                    self.stats['total_tokens'] += int(attention_mask.sum().item())

                    if torch.cuda.is_available() and (batch_idx + 1) % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

            for hook in hooks:
                hook.remove()

            self.logger.info(f"✓ Gram matrix accumulation completed (sequence-wise, pool='{pool}')")
            self.logger.info(f"  - Total batches: {total_batches}")
            self.logger.info(f"  - Total sequences (samples): {self.stats['total_samples']}")
            self.logger.info(f"  - Total valid tokens (참고): {self.stats['total_tokens']}")
            self.logger.info(f"  - Gram matrices: {len(self.gram_matrices)}")

            for key in sorted(self.gram_matrices.keys())[:3]:
                gram = self.gram_matrices[key]
                num = self.num_samples[key]
                self.logger.info(f"  - Layer {key}: Gram shape={tuple(gram.shape)}, samples(seq)={num}")
            if len(self.gram_matrices) > 3:
                self.logger.info(f"  - ... and {len(self.gram_matrices) - 3} more")

        except Exception as e:
            self.logger.error(f"Failed to accumulate Gram matrices (sequence-wise): {str(e)}", exc_info=True)
            raise

"""
SEAL data selector (TrainableTensorModule의 HF/PyTorch 이식).

SEAL의 selector σ(ω)는 길이 = downstream(선택 대상) 데이터셋 크기의 학습 가능한 logit 벡터다.
softmax면 합이 1이 되도록 정규화한 뒤 크기(size)를 곱해 평균 가중치≈1이 되게 하고,
sigmoid면 2를 곱한다(원본 SEAL과 동일한 norma).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableSelector(nn.Module):
    """per-sample 데이터 선택 가중치. forward()는 norma * activation(logits)."""

    def __init__(self, size: int, activation: str = "softmax"):
        super().__init__()
        if activation == "softmax":
            # 원본 SEAL: ones(size)*0.1 로 초기화
            self.logits = nn.Parameter((torch.ones(size) * 0.1).float())
            self.activ = nn.Softmax(dim=0)
            self.norma = size
        elif activation == "sigmoid":
            self.logits = nn.Parameter((-torch.ones(size) * 1e-4).float())
            self.activ = nn.Sigmoid()
            self.norma = 2
        else:
            raise ValueError(f"Unknown selector activation: {activation}")
        self.activation = activation
        self.size = size

    def forward(self) -> torch.Tensor:
        return self.norma * self.activ(self.logits)

    def negative_entropy(self) -> torch.Tensor:
        """softmax 전용. 선택 분포의 음의 엔트로피(정규화 항으로 사용 가능)."""
        logp = F.log_softmax(self.logits, dim=0, dtype=torch.float32)
        p = self.forward() / self.norma
        return (p * logp).sum()


def per_sample_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """
    per-sample 언어모델 CE 손실 벡터 (B,).

    각 샘플에 대해 유효(≠ignore_index) label 토큰의 평균 CE.
    선택 가중치를 샘플별로 곱하려면 배치 축소 이전의 per-sample 손실이 필요하다
    (SEAL의 batch_GPTLMLoss에 대응).
    """
    # shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Tm1, V = shift_logits.shape

    tok_loss = F.cross_entropy(
        shift_logits.view(-1, V).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, Tm1)

    valid = (shift_labels != ignore_index).float()
    per_sample = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return per_sample  # (B,)

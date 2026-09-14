"""SafeGrad: Gradient Surgery for Safe LLM Fine-tuning (arXiv:2508.07172).

이 패키지는 SafeGrad 를 이 저장소의 baseline 비교군으로 이식한 것이다.
자세한 내용은 safegrad/README.md 참고.
"""

from .safegrad_trainer import SafeGradTrainer  # noqa: F401

__all__ = ["SafeGradTrainer"]

"""
SEAL × WaRP 통합 실험 패키지.

SEAL (Safety-Enhanced Aligned LLM finetuning via Bilevel Data Selection, ICLR'25)의
data selector를 Safety-WaRP-LLM 저장소 스타일(HF Trainer + HF 모델)로 재구현하고,
선택된 downstream 데이터로 (1) 표준 full-param SFT(baseline)와
(2) WaRP 재매개변수화 공간 SFT를 각각 학습해 비교하기 위한 파이프라인.

파이프라인:
  Stage 1  train_selector.py : bilevel selector 학습
             - upper level = safe 데이터(circuit_breakers) 손실
             - lower level = selector 가중 downstream 데이터(gsm8k) 손실
             → selector logits(.pt) 저장
  Stage 1.5 select_data.py    : top-p% 임계 → 선택된 gsm8k 인덱스(json) 저장
  Stage 2  train_sft.py       : 선택된 gsm8k로 SFT
             --use_warp 없이 → baseline (전체 파라미터 학습)
             --use_warp 있이 → WaRP 공간 학습 (basis_coeff만 학습, 안전 방향 동결)
"""

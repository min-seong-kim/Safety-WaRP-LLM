# SEAL × WaRP 통합 실험

[SEAL](https://arxiv.org/abs/2410.07471)(bilevel 데이터 선택)의 data selector를 이 저장소
스타일(HF Trainer + HF 모델)로 재구현하고, 선택된 downstream 데이터로 **baseline SFT**와
**WaRP 공간 SFT**를 각각 학습해 비교한다.

- **Safety 데이터(upper level)** : `data/circuit_breakers_train.json` (`prompt` → `llama3_output` 거부 응답)
- **Downstream 데이터(선택 대상)** : `gsm8k` (`openai/gsm8k`, main/train)
- **베이스 모델** : `kmseong/llama2_7b-chat-Safety-FT-lr5e-5` (안전정렬)

## 비교 구도

| | 데이터 선택 | Stage-2 학습 |
|---|---|---|
| **(A) baseline** | SEAL selector top-p% | 표준 full-param SFT (전체 파라미터) |
| **(B) WaRP** | SEAL selector top-p% (동일) | WaRP 재매개변수화 공간, `basis_coeff`만 학습 (안전 방향 동결) |

두 경로는 데이터·토큰화·하이퍼파라미터를 완전히 공유하고 **WaRP 개입 여부만** 다르다.

## 파이프라인

```
Stage 1    train_selector.py  bilevel selector 학습          → ckpt/<name>_softmax.pt
Stage 1.5  select_data.py     top-p% 임계 선택               → ckpt/gsm8k_selected_topNN.json
Stage 2-A  train_sft.py       baseline SFT (선택 데이터)      → out/baseline_topNN
[WaRP 준비] train.py --phase 1 (basis, circuit_breakers)      → checkpoints/phase1_*/basis
           train.py --phase 2 (mask, keep_ratio)             → checkpoints/phase2_*/.../masks
Stage 2-B  train_sft.py --use_warp  WaRP SFT (선택 데이터)     → out/warp_topNN
```

## 한 번에 실행

```bash
# 저장소 루트에서
bash seal/scripts/run_all.sh
```

스크립트 상단 설정 블록(`CUDA_VISIBLE_DEVICES`, `MODEL`, `TOPP`, `KEEP_RATIO`, epoch/lr 등)을
환경에 맞게 수정한다. `PHASE1_BASIS_OVERRIDE`에 기존 basis 경로를 주면 Phase 1을 건너뛴다.

## 단계별 수동 실행

```bash
# Stage 1: selector 학습 (LoRA 권장 — 메모리 절약)
python -m seal.train_selector \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --safety_data_path data/circuit_breakers_train.json \
    --lora --epochs 2 --out_dir seal/ckpt --selector_name gsm8k_selector

# Stage 1.5: top-80% 선택
python -m seal.select_data \
    --selector_path seal/ckpt/gsm8k_selector_softmax.pt \
    --topp 0.8 --out seal/ckpt/gsm8k_selected_top80.json

# Stage 2-A: baseline
python -m seal.train_sft \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --selected_indices seal/ckpt/gsm8k_selected_top80.json \
    --output_dir seal/out/baseline_top80

# WaRP 준비 (Phase 1 basis + Phase 2 mask, 저장소 기존 train.py 사용)
python train.py --phase 1 --phase0_model_dir kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --safety_dataset circuit_breakers --circuit_breakers_samples_phase1 4994 \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up --target_layers all \
    --output_dir ./checkpoints --log_dir ./logs --device cuda --dtype bfloat16
python train.py --phase 2 --phase0_model_dir kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --basis_dir ./checkpoints/phase1_XXXX/basis \
    --circuit_breakers_path ./data/circuit_breakers_train.json \
    --dataset_phase2 circuit_breakers --circuit_breakers_samples_phase2 4994 \
    --keep_ratio 0.1 --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up --target_layers all \
    --output_dir ./checkpoints --log_dir ./logs --device cuda --dtype bfloat16 --perlayer

# Stage 2-B: WaRP
python -m seal.train_sft \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --selected_indices seal/ckpt/gsm8k_selected_top80.json \
    --use_warp \
    --basis_dir ./checkpoints/phase1_XXXX/basis \
    --masks_dir ./checkpoints/phase2_YYYY/checkpoints/masks \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up --target_layers all \
    --output_dir seal/out/warp_top80
```

## 불변식 / 주의

- **`--layer_type`, `--target_layers`는 Phase 1·2 및 WaRP SFT에서 반드시 동일**해야 한다
  (basis/mask/학습이 같은 (layer_idx, layer_type)를 인덱싱).
- selector logits(`.pt`)는 raw tensor다. `select_data.py`는 gsm8k train **순서 고정** 기준
  위치로 인덱싱하므로, Stage 1과 Stage 2의 `dataset_name/subset/split/num_train_samples`가
  동일해야 인덱스가 일치한다.
- WaRP는 frozen `basis_coeff`의 weight decay를 막기 위해 `weight_decay=0`을 강제한다 (Phase 3 규약).
- 토큰화는 `gsm8k_eval/finetune_gsm8k_full_params.py`와 동일 (instruct/chat → chat template,
  loss는 answer 토큰에만).

## 평가

이번 작업 범위는 **학습 파이프라인**까지다. 학습된 두 모델(`out/baseline_*`, `out/warp_*`)은
저장소의 기존 하네스로 평가한다:
- GSM8K 정확도 : `gsm8k_eval/`의 evaluate 스크립트
- 안전성(ASR/refusal) : `data/harmful_prompts_200.txt` 등 + 기존 safety 평가 스크립트

## 파일

- `data_utils.py` — 토큰화(gsm8k 스크립트와 동일), Dataset/Collator, gsm8k·circuit_breakers 빌더
- `selector.py` — `TrainableSelector`(softmax/sigmoid), `per_sample_lm_loss`
- `train_selector.py` — Stage 1 bilevel 학습 루프 (SEAL `SFTSelectorTrainer.fit` 이식)
- `select_data.py` — Stage 1.5 top-p% 선택
- `warp_setup.py` — Phase 3 `setup_warp_modules` 복제 (basis/mask 로드 + LinearWaRP 세팅 + 복원)
- `train_sft.py` — Stage 2 SFT (`--use_warp` 토글)
- `scripts/run_all.sh` — 전체 파이프라인
```

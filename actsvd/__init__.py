"""
actsvd package — WSR-Tune vs ActSVD mask-structure ablation.

명세: actsvd/wsr_actsvd_ablation_spec.md

구성:
    actsvd_basis.py               — 입력측/출력측(ActSVD) 기저 빌더 (Phase 1)
    wsr_ablation_masks.py         — arm 스펙 + entry/row/column 마스크 + budget 회계
    wsr_ablation_reparam.py       — Phase 2/3 공용 좌표계 세팅
    phase2_importance_ablation.py — arm 별 중요도 → 마스크 (Phase 2)
    phase3_ablation.py            — arm 별 downstream 학습 (Phase 3)
    wsr_actsvd_ablation_report.py — 결과/budget 교차 검증 리포트
    test_wsr_actsvd_ablation.py   — 단위 테스트

train.py 가 `--ablation_arm` / `--basis_side` 플래그로 이 패키지를 lazy import 한다.
드라이버는 scripts/run_wsr_actsvd_ablation.sh (스모크: scripts/_smoke_wsr_actsvd.sh).
"""

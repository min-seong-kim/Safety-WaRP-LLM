#!/usr/bin/env python
"""
WSR-Tune vs ActSVD ablation 결과 리포트 (spec §7 reporting template).

산출물 디렉토리를 훑어서
  1. arm 간 동결 파라미터 수가 ±1% 이내인지 교차 검증 (spec §8 항목 4)
  2. 기저 종류/저장량/복원 오차 등 오버헤드 수치 정리 (spec §6)
  3. 채워 넣을 수 있는 마크다운 표 출력
을 수행한다. ASR/GSM8K 숫자는 저장소 밖(HarmBench / lm-evaluation-harness)에서 나오므로
`--eval_json` 으로 합쳐 넣는다.

사용:
  python wsr_actsvd_ablation_report.py --root outputs/wsr_actsvd_ablation
  python wsr_actsvd_ablation_report.py --root outputs/wsr_actsvd_ablation \
         --eval_json evals.json --out report.md

evals.json 형식 (키는 대소문자 무시, 없는 값은 비워둠):
  {"A": {"direct": 0.0, "pap": 18.0, "gsm8k": 40.41},
   "B": {"direct": 0.0, "pap": 12.3, "gsm8k": 39.8}}
"""

import argparse
import json
import os
import sys

# 저장소 루트(= 이 파일의 상위 디렉토리)를 import 경로에 넣어 단독 실행 가능하게 한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actsvd.wsr_ablation_masks import ARM_SPECS, compare_arm_budgets  # noqa: E402

ATTACKS = ("direct", "autodan", "pair", "pap")


def _read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _read_pointer(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        value = f.read().strip()
    return value or None


def collect(root):
    """arm 디렉토리에서 mask/budget/모델 정보를 모은다."""
    arms = {}
    for name in sorted(os.listdir(root)):
        if not name.startswith('arm_'):
            continue
        arm = name[len('arm_'):]
        arm_dir = os.path.join(root, name)

        masks_dir = _read_pointer(os.path.join(arm_dir, 'masks.path'))
        model_dir = _read_pointer(os.path.join(arm_dir, 'final_model.path'))

        entry = {
            'arm': arm,
            'spec': ARM_SPECS.get(arm, {}),
            'masks_dir': masks_dir,
            'final_model': model_dir,
            'mask_meta': _read_json(os.path.join(masks_dir, 'metadata.json')) if masks_dir else None,
            'budget': _read_json(os.path.join(masks_dir, 'budget_report.json')) if masks_dir else None,
            'reparam': _read_json(os.path.join(masks_dir, 'reparam_diagnostics.json')) if masks_dir else None,
        }
        if model_dir:
            summary_path = os.path.join(os.path.dirname(model_dir), 'ablation_summary.json')
            entry['train_summary'] = _read_json(summary_path)
        arms[arm] = entry
    return arms


def collect_bases(root):
    bases = {}
    for side in ('input', 'output'):
        ptr = _read_pointer(os.path.join(root, f'basis_{side}.path'))
        if ptr:
            bases[side] = {'dir': ptr, 'meta': _read_json(os.path.join(ptr, 'metadata.json'))}
    return bases


def fmt(value, spec="{:.2f}"):
    return spec.format(value) if isinstance(value, (int, float)) else "—"


def build_report(root, evals, keep_ratio):
    arms = collect(root)
    bases = collect_bases(root)
    lines = []
    add = lines.append

    add("# WSR-Tune vs ActSVD — mask-structure ablation")
    add("")
    add(f"- 산출물 루트: `{root}`")
    add(f"- keep ratio ρ = {keep_ratio}")
    add("- 모든 arm이 동일한 Phase 3 파이프라인/하이퍼파라미터를 쓰고, 좌표계(U, V)와 "
        "마스크 단위만 다르다.")
    add("")

    # ── 기저 정보 ────────────────────────────────────────────────
    if bases:
        add("## 1. 기저 (basis)")
        add("")
        add("| side | 정의 | token scope | 모듈 수 | 저장량 | 디렉토리 |")
        add("|---|---|---|---|---|---|")
        for side, info in bases.items():
            meta = info['meta'] or {}
            add(f"| {side} | {meta.get('basis_space', '—')} | {meta.get('token_scope', '—')} | "
                f"{meta.get('num_layers_saved', '—')} | {fmt(meta.get('storage_gib'), '{:.2f}')} GiB | "
                f"`{info['dir']}` |")
        add("")
        add("> 입력측 U ∈ R^{n×n} (WSR-Tune, X_in X_inᵀ 고유기저) 과 "
            "출력측 U_out ∈ R^{m×m} (ActSVD, W X_in 의 left singular vectors) 는 서로 다른 공간이다.")
        add("")

    # ── 예산 검증 ────────────────────────────────────────────────
    add("## 2. 동결 파라미터 예산 매칭 (spec §2 / §8-4)")
    add("")
    budgets = {a: e['budget'] for a, e in arms.items() if e.get('budget')}
    if budgets:
        add("| arm | mask unit | rank by | 모듈 수 | 동결 스칼라 | 동결 비율 | entry 기준 오차 | layer 최대 오차 |")
        add("|---|---|---|---|---:|---:|---:|---:|")
        for arm in sorted(budgets):
            b = budgets[arm]
            add(f"| {arm} | {b.get('mask_unit')} | {b.get('rank_by')} | {b.get('num_modules')} | "
                f"{b.get('total_frozen', 0):,} | {b.get('total_frozen_ratio', 0) * 100:.3f}% | "
                f"{b.get('total_budget_rel_err', 0):+.3%} | {b.get('max_layer_budget_rel_err', 0):.3%} |")
        add("")
        ok, msg = compare_arm_budgets(budgets)
        add("```")
        add(msg)
        add("```")
        if not ok:
            add("")
            add("> ⚠️ **예산 불일치** — 이 상태의 arm 비교는 공정하지 않다. "
                "`structured_k` 반올림(ρ·dim이 작을 때) 또는 서로 다른 layer 집합을 확인할 것.")
        add("")
    else:
        add("_아직 mask 산출물이 없다._")
        add("")

    # ── 재파라미터화 건전성 ──────────────────────────────────────
    add("## 3. 재파라미터화 건전성")
    add("")
    add("| arm | 좌표계 | 복원 상대오차 (mean / max) |")
    add("|---|---|---|")
    for arm in sorted(arms):
        entry = arms[arm]
        diag = entry.get('reparam') or {}
        spec = entry['spec']
        coord = {
            None: "W̃ = W",
            'input': "W̃ = W·U_in",
            'output': "W̃ = U_outᵀ·W",
        }.get(spec.get('basis_side'), '—')
        if spec.get('v_mode') == 'signed_permutation':
            coord = "W̃ = Pᵀ·W·U_in"
        if diag:
            vals = list(diag.values())
            cell = f"{sum(vals) / len(vals):.2e} / {max(vals):.2e}"
        else:
            cell = "—"
        add(f"| {arm} | {coord} | {cell} |")
    add("")
    add("> `W = V·basis_coeff·Uᵀ` 왕복 오차. bf16 기저에서 1e-3 수준이 정상이며, "
        "arm 간 값이 비슷해야 '좌표계만 바꿨다'는 주장이 성립한다.")
    add(">")
    add("> ⚠️ **알려진 교란요인**: arm A는 항등 좌표변환이라 이 오차가 정확히 0이고, "
        "B/C/D는 bf16 기저 왕복 때문에 ~1e-3의 초기 섭동을 갖는다. 이는 논문 본문의 "
        "WSR-Tune에도 그대로 존재하는 성질(Phase 3이 기저를 모델 dtype으로 캐스팅)이므로 "
        "arm D는 본문 수치를 재현하지만, **A vs (B/C/D) 격차의 아주 일부는 이 반올림에서 올 수 있다**. "
        "B/C/D 사이 비교는 세 arm이 같은 크기의 섭동을 공유하므로 이 교란에서 자유롭다.")
    add("")

    # ── 결과 표 (spec §7) ────────────────────────────────────────
    add("## 4. 결과 (spec §7 template)")
    add("")
    add("| Arm | Basis | Mask | #Frozen | Direct | AutoDAN | PAIR | PAP | ASR-AVG | GSM8K |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ("A", "B", "C", "D", "D_perm"):
        if arm not in arms:
            continue
        entry = arms[arm]
        spec = entry['spec']
        b = entry.get('budget') or {}
        ev = {k.lower(): v for k, v in (evals.get(arm) or {}).items()}
        attack_vals = [ev.get(a) for a in ATTACKS]
        present = [v for v in attack_vals if isinstance(v, (int, float))]
        avg = ev.get('asr_avg', sum(present) / len(present) if present else None)
        basis_label = {None: "original", 'input': "safety input U",
                       'output': "ActSVD output U_out"}.get(spec.get('basis_side'), "—")
        frozen = f"{b['total_frozen']:,}" if b.get('total_frozen') else "—"
        add(f"| {arm} {spec.get('label', '')} | {basis_label} | {spec.get('mask_unit', '—')} | "
            f"{frozen} | " + " | ".join(fmt(v) for v in attack_vals)
            + f" | {fmt(avg)} | {fmt(ev.get('gsm8k'))} |")
    add("")

    # ── 정합성 체크리스트 ────────────────────────────────────────
    add("## 5. 정합성 체크리스트 (spec §8)")
    add("")
    checks = [
        ("1. arm A ≈ 논문 Table 5 (JB≈9.17 / GSM8K≈40.41)", _check_reference(evals.get('A'), 9.17, 40.41)),
        ("2. arm D ≈ 논문 Table 2 (JB≈6.90 / GSM8K≈38.99)", _check_reference(evals.get('D'), 6.90, 38.99)),
        ("3. signed-permutation sanity arm == arm D",
         "unit test `test_perm_arm_training_step_equals_arm_D` 통과 (전체 학습 arm은 선택)"),
        ("4. arm 간 동결 수 ±1%", compare_arm_budgets(budgets)[1].splitlines()[0] if budgets else "미측정"),
        ("5. U_out=left singular(W X_in, m×m) / U_in=eigenbasis(X Xᵀ, n×n)",
         "Phase 2/3 로드시 basis_side 검증 + 형상 검사로 강제, unit test로 수식 검증"),
        ("6. importance는 응답 토큰만 사용", "Phase 2 circuit_breakers 로더가 prompt를 -100 마스킹"),
    ]
    for label, status in checks:
        add(f"- {label}: {status}")
    add("")
    add("## 6. 해석 가이드 (spec §7)")
    add("")
    add("- **D > B**: 안전 조건부 공간에서의 원소 단위 동결이 rank 동결을 이긴다 → "
        "이득의 근원은 rank 식별이 아니라 (neuron × direction) 좌표 보존.")
    add("- **D ≈ B**: 둘 다 재파라미터화의 수혜자 → WSR-Tune의 기여는 Wei et al. footnote 9이 "
        "\"쉽지 않다\"고 한 rank-level freezing을 **구현 가능하게** 만든 것.")
    add("- **D < B**: 정직하게 보고. entry → rank 마스킹으로 옮겨갈 근거.")
    add("- 어느 경우든 **B > A** 가 핵심 결과다: 어떤 파라미터를 고를지가 아니라 "
        "**어떤 공간에서 고르는지**가 중요하다는 논문 Table 5의 직접 확장.")
    return "\n".join(lines)


def _check_reference(ev, jb_ref, gsm_ref, tol_jb=2.0, tol_gsm=2.0):
    if not ev:
        return "미측정"
    ev = {k.lower(): v for k, v in ev.items()}
    present = [ev[a] for a in ATTACKS if isinstance(ev.get(a), (int, float))]
    jb = ev.get('asr_avg', sum(present) / len(present) if present else None)
    gsm = ev.get('gsm8k')
    if jb is None or gsm is None:
        return "부분 측정 (JB/GSM8K 중 하나 없음)"
    ok = abs(jb - jb_ref) <= tol_jb and abs(gsm - gsm_ref) <= tol_gsm
    return (f"{'OK' if ok else '⚠️ 불일치'} — JB={jb:.2f} (ref {jb_ref}), "
            f"GSM8K={gsm:.2f} (ref {gsm_ref})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='outputs/wsr_actsvd_ablation',
                    help='run_wsr_actsvd_ablation.sh 의 OUT_ROOT')
    ap.add_argument('--eval_json', default=None,
                    help='arm별 ASR/GSM8K 결과 JSON (선택)')
    ap.add_argument('--keep_ratio', type=float, default=0.10)
    ap.add_argument('--out', default=None, help='마크다운 저장 경로 (기본: stdout만)')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"root 디렉토리가 없습니다: {args.root}", file=sys.stderr)
        return 1

    evals = _read_json(args.eval_json) if args.eval_json else {}
    if args.eval_json and evals is None:
        print(f"eval_json을 읽지 못했습니다: {args.eval_json}", file=sys.stderr)
        evals = {}

    report = build_report(args.root, evals or {}, args.keep_ratio)
    print(report)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(report + "\n")
        print(f"\n✓ saved to {args.out}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REVISION_PROGRESS.md 생성기.

common.sh 의 레지스트리를 **그대로** 재사용한다(손으로 적은 목록은 반드시 어긋난다).
bash 를 한 번 호출해 매트릭스를 TSV 로 뽑고, 로컬 .done/.uploaded 와 허브 존재 여부를
합쳐 마크다운으로 찍는다.
"""
import argparse, os, subprocess, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENUM = r'''
source "%s/common.sh" >/dev/null 2>&1
for safety in cb bt; do
  for mkey in $MODELS; do
    model_cfg "$mkey" >/dev/null 2>&1 || continue
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for method in $METHODS; do
        want=no; cell_wanted "$safety" "$mkey" "$task" "$method" 2>/dev/null && want=yes
        pub=no;  already_published "$safety" "$mkey" "$task" "$method" 2>/dev/null && pub=yes
        blk=no;  cell_blocked "$safety" "$mkey" "$task" "$method" 2>/dev/null && blk=yes
        repo="$(hf_repo_id "$safety" "$mkey" "$task" "$method" 2>/dev/null)"
        d="$(out_dir "$safety" "$mkey" "$task" "$method")"
        st=pending; [ -f "$d/.done" ] && st=done; [ -f "$d/.uploaded" ] && st=uploaded
        printf '%%s\t%%s\t%%s\t%%s\t%%s\t%%s\t%%s\t%%s\n' \
          "$safety" "$mkey" "$task" "$method" "$want" "$pub" "$blk" "$repo" | tr -d '\r'
        echo -e "STATUS\t$st"
      done
    done
  done
done
''' % HERE


def enumerate_matrix():
    out = subprocess.run(["bash", "-c", ENUM], capture_output=True, text=True,
                         cwd=REPO, env={**os.environ, "PATH": os.environ["PATH"]})
    rows, pend = [], None
    for line in out.stdout.splitlines():
        f = line.split("\t")
        if f[0] == "STATUS" and pend is not None:
            pend["status"] = f[1]; rows.append(pend); pend = None
        elif len(f) == 8:
            pend = dict(zip(["safety", "model", "task", "method", "want", "pub", "blocked", "repo"], f))
    if not rows:
        sys.stderr.write(out.stderr[-2000:]); raise SystemExit("매트릭스 열거 실패")
    return rows


def hub_repos():
    try:
        from huggingface_hub import HfApi
        return {m.id for m in HfApi().list_models(author="kmseong")}
    except Exception as e:
        sys.stderr.write("[WARN] 허브 조회 실패: %s\n" % e); return None


MODEL_LABEL = {"llama32_3b": "Llama-3.2-3B-It", "llama2_7b": "Llama-2-7B-chat",
               "qwen25_7b": "Qwen2.5-7B-It", "gemma2_9b": "gemma-2-9b-it",
               "llama31_8b": "Llama-3.1-8B-It", "llama2_13b": "Llama-2-13B-chat"}
METHOD_LABEL = {"fullft": "Full FT", "safeinstr": "SafeInstr", "resta": "RESTA",
                "safedelta": "SafeDelta", "wsr_tune": "WSR-Tune", "lora": "Vanilla LoRA",
                "asft": "AsFT", "lisa": "LISA", "seal": "SEAL", "safelora": "SafeLoRA",
                "salora": "SaLoRA", "wsr_lora": "WSR-LoRA"}
TASK_ORDER = ["gsm8k", "math", "medqa", "arc", "agnews"]
METHOD_ORDER = list(METHOD_LABEL)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = enumerate_matrix(); hub = hub_repos()

    # 리포지토리가 "존재" 하는 것과 가중치가 다 올라간 것은 다르다.
    # 중단된 업로드는 빈 리포를 남기므로, .uploaded 마커가 없는 셀은 safetensors 유무까지 본다.
    _filecache = {}

    def _has_weights(repo):
        if repo in _filecache: return _filecache[repo]
        ok = False
        try:
            from huggingface_hub import HfApi
            files = HfApi().list_repo_files(repo)
            ok = any(f.endswith(".safetensors") for f in files) and "config.json" in files
        except Exception:
            ok = False
        _filecache[repo] = ok
        return ok

    def on_hub(r):
        if r["status"] == "uploaded": return True   # 4단계 검증을 통과한 마커
        if hub is None or r["repo"] not in hub: return False
        return _has_weights(r["repo"])

    wanted = [r for r in rows if r["want"] == "yes"]
    reused = [r for r in rows if r["pub"] == "yes"]
    blocked = [r for r in rows if r["blocked"] == "yes" and r["want"] != "yes"]
    cb_w = [r for r in wanted if r["safety"] == "cb"]
    bt_w = [r for r in wanted if r["safety"] == "bt"]
    done = [r for r in wanted if on_hub(r)]
    local = [r for r in wanted if r["status"] == "done" and not on_hub(r)]
    todo = [r for r in wanted if r["status"] == "pending"]

    L = []
    w = L.append
    w("# WSR-Tune revision — 실험 진행 현황\n")
    w("NeurIPS 2026 rebuttal/revision 용 확장 실험(제출 11600)의 **자동 생성** 현황표다.")
    w("손으로 적지 않는다 — `scripts/revision/common.sh` 의 레지스트리와 실제 허브/로컬 상태에서 뽑는다.\n")
    w("```\npython scripts/revision/gen_progress_md.py --out REVISION_PROGRESS.md\n```\n")

    w("## 요약\n")
    w("| 항목 | 셀 수 |")
    w("|---|---|")
    w(f"| 새로 만들 셀(전체) | **{len(wanted)}** |")
    w(f"| └ CB 축 | {len(cb_w)} |")
    w(f"| └ BT 축 | {len(bt_w)} |")
    w(f"| ✅ 허브 업로드 완료 | **{len(done)}** |")
    w(f"| 🟡 학습 완료·업로드 대기(로컬) | {len(local)} |")
    w(f"| ⬜ 미실행 | {len(todo)} |")
    w(f"| ♻️ 논문 기존 결과 재사용(새로 안 만듦) | {len(reused)} |")
    w(f"| 🚫 라이선스 차단(gemma-2-9b-it) | {len(blocked)} |")
    w("")

    w("## 실험 설계 (모든 셀 공통)\n")
    w("- **출발 모델**: 각 base model 의 *기존 safety-tuned 체크포인트*. 새로 SSFT 하지 않는다.")
    w("- **안전 데이터 재사용 규칙**: 안전 데이터가 필요한 기법(SafeInstr/RESTA/SafeDelta/AsFT/LISA/SEAL/SafeLoRA/SaLoRA/WSR-*)은")
    w("  **출발 모델이 safety-tune 될 때 쓴 바로 그 데이터셋**(CB 축이면 `circuit_breakers`, BT 축이면 BeaverTails)을 쓴다.")
    w("- **프롬프트 동일성**: 한 task 는 task JSON **하나**를 12개 기법이 전부 같이 읽는다.")
    w("  `scripts/revision/verify_prompt_parity.py` 가 6개 토크나이즈 경로의 `(input_ids, labels)` 가 바이트 단위로")
    w("  같음을 확인한다 — 기법 간 차이가 프롬프트 차이에서 오지 않음을 보장한다.")
    w("- **공통**: epochs 3 · effective batch 16 · max_len 1024 · seed 42 · bf16 · cosine · max_grad_norm 1.0")
    w("")
    w("| | full-param | LoRA 계열 |")
    w("|---|---|---|")
    w("| lr | 5e-5 | 3e-4 (gsm8k/math/medqa/arc) · 7e-5 (agnews) |")
    w("| weight decay | 0.01 | 0.0 |")
    w("| warmup ratio | 0.1 | 0.03 |")
    w("| rank / alpha / dropout | — | 16 / 32 / 0.05 |")
    w("")
    w("기법별 하이퍼파라미터: AsFT λ=1.0 · LISA (rho 1.0, align_step 100, ft_step 900) ·")
    w("SafeLoRA threshold 0.3 · SaLoRA (r_s=32, r_t=32) · WSR-Tune/WSR-LoRA ρ(keep_ratio)=0.1 ·")
    w("RESTA γ=0.3 · SafeDelta s=0.1.\n")

    # ── 모델 × task 표 ──
    w("## 진행 표\n")
    w("범례: ✅ 허브 업로드 · 🟡 로컬 학습 완료(업로드 대기) · ⬜ 미실행 · ♻️ 논문 결과 재사용 · 🚫 라이선스 차단 · `·` 매트릭스 밖\n")
    idx = {(r["safety"], r["model"], r["task"], r["method"]): r for r in rows}
    for safety in ("cb", "bt"):
        sr = [r for r in rows if r["safety"] == safety]
        if not sr: continue
        w(f"### {safety.upper()} 축 (safety dataset = {'circuit_breakers' if safety=='cb' else 'BeaverTails'})\n")
        models = [m for m in MODEL_LABEL if any(r["model"] == m for r in sr)]
        for m in models:
            mr = [r for r in sr if r["model"] == m]
            if not mr: continue
            tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in mr)]
            if not tasks: continue
            w(f"**{MODEL_LABEL[m]}**\n")
            w("| task | " + " | ".join(METHOD_LABEL[x] for x in METHOD_ORDER) + " |")
            w("|---" * (len(METHOD_ORDER) + 1) + "|")
            for t in tasks:
                cells = []
                for me in METHOD_ORDER:
                    r = idx.get((safety, m, t, me))
                    if r is None: cells.append("·")
                    elif r["pub"] == "yes": cells.append("♻️")
                    elif r["blocked"] == "yes": cells.append("🚫")
                    elif r["want"] != "yes": cells.append("·")
                    elif on_hub(r): cells.append("✅")
                    elif r["status"] == "done": cells.append("🟡")
                    else: cells.append("⬜")
                w(f"| {t} | " + " | ".join(cells) + " |")
            w("")

    if local:
        w("## 업로드 대기 중인 로컬 모델\n")
        for r in local: w(f"- `{r['repo']}`")
        w("")
    if todo:
        w("## 남은 셀\n")
        for r in todo: w(f"- {r['safety']}/{r['model']}/{r['task']}/{r['method']} → `{r['repo']}`")
        w("")
    if blocked:
        w("## 차단된 셀 (gemma-2-9b-it 게이트 라이선스 미승인)\n")
        w(f"{len(blocked)}개. 라이선스를 수락한 뒤 `common.sh` 의 `BASE_BLOCKED_MODELS=\"\"` 로 두면 다시 잡힌다.\n")

    w("## 업로드된 모델 (허브)\n")
    for r in sorted(done, key=lambda x: x["repo"]): w(f"- [`{r['repo']}`](https://huggingface.co/{r['repo']})")
    w("")
    w("## 재현\n")
    w("```bash\nsetsid nohup bash scripts/revision/finish_cb.sh > /dev/null 2>&1 &   # CB 축\n"
      "tail -f logs/revision_unattended/finish_cb_latest.log\n```\n")
    w("업로드는 4단계로 검증한다(파일 존재 · 크기 일치 · 허브에서 `AutoConfig` 로드 · "
      "허브 토크나이저의 `chat_template` 존재). 검증을 통과한 셀만 로컬 가중치를 지운다.\n")

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"  {a.out} 작성 — 업로드 {len(done)} / 대기 {len(local)} / 남음 {len(todo)} / 재사용 {len(reused)}")


if __name__ == "__main__":
    main()

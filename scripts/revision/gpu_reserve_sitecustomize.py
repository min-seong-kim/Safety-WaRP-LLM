# 공유 GPU 대책 (2026-09-06): 학습 프로세스 시작 직후 큰 블록을 잡았다 놓아 caching allocator 에 예약해 둔다.
#   다른 사용자의 idle-wait 스크립트가 우리 피크 도달 전(모델 로드~importance 역전파, 수 분) 메모리를 가져가 OOM 나는 것을 막는다.
#   제어 파일: ~/Safety-WaRP-LLM/logs/revision_sweep/RESERVE_GPU_GIB  (없거나 0 이면 아무것도 안 함)
#   대상: wsr_lora.py / finetune_gsm8k_lora.py / train.py 만. 수치·하이퍼파라미터에는 영향 없음(메모리 예약만).
import os, sys
try:
    _ctl = os.path.expanduser("~/Safety-WaRP-LLM/logs/revision_sweep/RESERVE_GPU_GIB")
    _argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    if os.path.isfile(_ctl) and _argv0 in ("wsr_lora.py", "finetune_gsm8k_lora.py", "train.py"):
        _cfg = {}
        for _line in open(_ctl):
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                k, v = _line.split("=", 1); _cfg[k.strip()] = float(v)
        _args = " ".join(sys.argv).lower()
        _model = "gemma" if "gemma" in _args else ("13b" if "13b" in _args else "other")
        _kind = "wsr" if _argv0 == "wsr_lora.py" else "ft"
        _gib = _cfg.get(f"{_model}_{_kind}", _cfg.get("default", 0.0))
        if _gib > 0:
            import torch
            _free, _total = torch.cuda.mem_get_info()
            _want = min(_gib, _free / 2**30 - 1.5)
            if _want > 4:
                _x = torch.empty(int(_want * 2**30), dtype=torch.uint8, device="cuda"); del _x
                # wsr_lora.py 등이 단계 사이에 torch.cuda.empty_cache() 를 불러 예약이 풀린다(2026-09-06 실측) → 무력화.
                #   캐시를 OS 에 돌려주지 않을 뿐 계산 결과에는 영향이 없다.
                torch.cuda.empty_cache = lambda *a, **k: None
                print(f"[sitecustomize] GPU 메모리 {_want:.1f} GiB 예약 (free was {_free/2**30:.1f}, target {_gib}); empty_cache 무력화", flush=True)
            else:
                print(f"[sitecustomize] 예약 생략: free {_free/2**30:.1f} GiB", flush=True)
except Exception as _e:
    print(f"[sitecustomize] 예약 실패(무시): {_e!r}", flush=True)

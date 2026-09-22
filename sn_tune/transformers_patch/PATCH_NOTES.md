# 패치된 transformers modeling 파일

Safety-Neuron 논문(ICLR'25, *Understanding and Enhancing Safety Mechanisms of LLMs via
Safety-Specific Neuron*)의 **원공간 뉴런 검출**은 정품 `transformers` 로는 돌아가지 않는다.
검출 스크립트는 forward 중에 모듈에 스태시된 중간 점수를 읽는데, 그 스태시를 심는 것이
이 디렉토리의 파일들이다.

| 파일 | 설치 위치 (site-packages 기준) |
|---|---|
| `modeling_llama.py`  | `transformers/models/llama/modeling_llama.py` |
| `modeling_gemma2.py` | `transformers/models/gemma2/modeling_gemma2.py` |
| `modeling_qwen2.py`  | `transformers/models/qwen2/modeling_qwen2.py` |

패치가 심는 속성 (검출기가 `getattr` 로 읽는다):

```
layer.mlp._last_ffn_up_score      layer.mlp._last_ffn_down_score
layer.self_attn._last_q_score     layer.self_attn._last_k_score
layer.self_attn._last_v_score     layer.self_attn._last_o_score
layer.self_attn._last_query_states / _last_key_states / _last_value_states
```

## 출처와 해시

원본: `Safety-Neuron/neuron_detection/transformers/models/` 의
`modeling_llama (2).py` · `modeling_gemma2 (1).py` · `modeling_qwen2 (1).py`
(공백과 `(N)` 가 붙은 그 파일들이 맞다. 같은 트리의 `models/llama/modeling_llama.py`
같은 **서브패키지 사본은 훨씬 오래된 스냅샷이라 쓰면 안 된다** — 1687줄 vs 584줄).

2026-09-22 에 바이트 그대로 들여왔다:

| 파일 | sha256 | 줄수 | 줄끝 |
|---|---|---:|---|
| `modeling_llama.py`  | `1b064bf1ed25a21ce608abad05427f9d28b3224bcce0bb73823f67353db0285a` | 584 | LF |
| `modeling_gemma2.py` | `a51601042fbe8de7af41b3d1e57d7780e0530dc60ba0b0e42aae22967f5e83a2` | 706 | **CRLF** |
| `modeling_qwen2.py`  | `92ffc96a33cca00827765a683b9fd0dceef07c109bf14002145df5fa85f53ce9` | 635 | **CRLF** |

대응하는 transformers **4.57.3 정품** 파일 sha256 (이게 나오면 패치가 **안** 걸린 것):

| 모델 | 정품 sha256 | 줄수 |
|---|---|---:|
| llama  | `f4dcfcb9468579f94fce84ea5740d69a8c7934df936070aff863375447ddcabb` | 505 |
| gemma2 | `fa77bf171148bb47ba4a53ebca02f4e6cdcdcda15ea2e23aea4a296539bd203c` | 598 |
| qwen2  | `08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8` | 498 |

## 함정

- **줄끝이 CRLF 다** (gemma2·qwen2). `sed -i 's/...$/.../'` 처럼 `$` 로 앵커를 잡는 패턴은
  `\r` 때문에 **조용히 안 맞는다**. 토큰으로 앵커를 잡거나 `\r?$` 를 쓸 것.
- **패치는 설치된 것보다 오래된 transformers 를 겨냥한다.** 4.57.3 에서 발견·수정된 사례:
  `modeling_qwen2.py` 의 `@check_model_inputs` → 4.57.3 에서는 이 데코레이터가
  `tie_last_hidden_states` 를 받는 **팩토리**가 되어서, 괄호 없는 형태는
  `Qwen2Model.forward` 를 내부 `wrapped_fn` 으로 바꿔버리고 모든 forward 가
  `TypeError: wrapped_fn() got an unexpected keyword argument 'input_ids'` 로 죽는다.
  `@check_model_inputs()` 로 고쳐져 있다. **다른 계열에도 비슷한 드리프트가 잠복해 있다고
  가정할 것** — transformers 를 올리거나 새 계열을 추가하면 그 버전의 정품과 먼저 diff 하라.
  빠른 확인: `inspect.signature(XModel.forward)` 가 `['self','input_ids',...]` 로 시작해야지
  `['func']` 면 안 된다. `verify_patch.py` 가 이걸 검사한다.
- **학습 환경에는 절대 깔지 마라.** 패치는 검출 전용이다. 학습(`hb`)은 정품이어야 한다.
  `setup_hb_sn.sh` 는 `hb` 를 건드리지 않고 `hb_sn` 만 만든다.

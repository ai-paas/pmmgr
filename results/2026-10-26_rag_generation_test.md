# Chat model generation test

- Base URL: `http://101.202.128.10:8001`
- Query: `Tell me about RAG briefly`
- Passes per model: 2
- Models: 20/20 generated a non-empty answer
- Calls: 40 (total 275.1s, median 3.205s, min 0.61s, max 92.95s)
- Answer size: median 1308 chars, min 339, max 75166
- Verdict: HTTP 200 with an empty body counts as a failure.
- Answer bodies are not stored in this report; pass --json-out for raw output.

| # | model | run1 | run2 | 생성 |
|---|-------|---|---|---|
| 1 | `deepseek-r1:1.5b` | 1.18s / 1473자 | 1.42s / 1968자 | ✅ |
| 2 | `gemma3:1b` | 2.78s / 1975자 | 3.22s / 2424자 | ✅ |
| 3 | `llama3.2:1b` | 2.56s / 684자 | 1.0s / 1233자 | ✅ |
| 4 | `qwen2.5:0.5b` | 7.37s / 1301자 | 1.08s / 1573자 | ✅ |
| 5 | `phi3:3.8b` | 7.81s / 2579자 | 1.38s / 1743자 | ✅ |
| 6 | `qwen2.5-coder:0.5b` | 7.45s / 1430자 | 0.83s / 1060자 | ✅ |
| 7 | `tinyllama:1.1b` | 6.23s / 967자 | 0.75s / 2245자 | ✅ |
| 8 | `starcoder2:3b` | 92.95s / 75166자 | 0.61s / 339자 | ✅ |
| 9 | `granite3.1-moe:1b` | 6.73s / 656자 | 0.64s / 1255자 | ✅ |
| 10 | `falcon3:1b` | 3.19s / 1798자 | 1.07s / 1470자 | ✅ |
| 11 | `lfm2.5:8b` | 4.75s / 907자 | 2.27s / 1612자 | ✅ |
| 12 | `medgemma1.5:4b` | 15.26s / 4659자 | 7.5s / 4759자 | ✅ |
| 13 | `minicpm-v4.6:1b` | 4.66s / 392자 | 1.39s / 561자 | ✅ |
| 14 | `ornith-1.5:9b` | 9.57s / 1261자 | 4.84s / 1096자 | ✅ |
| 15 | `translategemma:4b` | 11.44s / 1971자 | 2.35s / 1316자 | ✅ |
| 16 | `nemotron-3-nano:4b` | 4.57s / 1060자 | 1.5s / 828자 | ✅ |
| 17 | `qwen3-vl:2b` | 11.59s / 1860자 | 3.86s / 1966자 | ✅ |
| 18 | `rnj-1:8b` | 8.92s / 832자 | 1.29s / 817자 | ✅ |
| 19 | `olmo-3:7b` | 17.58s / 1090자 | 8.15s / 2551자 | ✅ |
| 20 | `granite4:3b` | 2.51s / 1007자 | 0.81s / 845자 | ✅ |

## 관찰

- `starcoder2:3b`는 run1에서 75,166자 / 92.95s까지 생성됐고 run2는 339자였다. API에 출력 길이 상한(`num_predict`, `stop`)이 없어 샘플링에 따라 폭주 폭이 크게 달라진다.
- `medgemma1.5:4b`는 두 회차 모두 응답 선두에 `<unused94> thought` 특수토큰이 그대로 노출된다. `pmmgr/main.py`가 `result.content`를 후처리 없이 반환하기 때문이다.
- 2회차가 1회차보다 빠르다(중앙값 7.05s → 1.39s, 40회 전체 중앙값은 3.21s). 원인은 확인하지 않았다(백엔드 프롬프트 캐시 또는 짧아진 출력 길이 가능성).


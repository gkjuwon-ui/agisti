# AGISTI Phase 1: GPU Surgery Training Report 🧠⚡

## 실행 환경

| 항목 | 값 |
|------|-----|
| **GPU** | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| **Pod** | RunPod `giant_aqua_mole` |
| **PyTorch** | 2.12.0.dev20260313+cu128 (Nightly) |
| **Python** | 3.11.10 |
| **CUDA** | 12.8 (cu128) |
| **모델** | Qwen/Qwen2.5-3B (3,085.9M params, bfloat16) |
| **GPU 메모리** | 6.2GB (로드 시) → 13.2GB (피크) |

## RTX 5090 호환성 삽질기 🔧

RTX 5090은 **Blackwell 아키텍처 (sm_120)** — 2026년 3월 기준으로도 공식 지원하는 PyTorch가 없었다.

1. **기본 설치된 PyTorch 2.4.1+cu124**: sm_90까지만 지원 → `no kernel image` 에러
2. **Nightly cu126 빌드**: 다운은 되는데 역시 sm_120 미지원 → 같은 에러
3. **Nightly cu128 빌드**: ✅ 드디어 작동! `torch-2.12.0.dev20260313+cu128`

> RTX 5090 쓰려면 **반드시 cu128 이상** 필요하다. cu126까지는 Blackwell 커널이 없음.

## Phase 1 실행 결과

### 설정
```
--iterations 2
--epoch-size 2
--skip-frozen
--model Qwen/Qwen2.5-3B
--bench-problems 5
--probes-per-domain 3
--lora-rank 4
```

### Iteration 0
| 단계 | 결과 |
|------|------|
| **Probe** | overall 13.3% (logic 0%, knowledge 33%, math 0%, reading 33%, coding 0%) |
| **Generation** | 10 problems (10 verifiable) |
| **Evaluation** | 0/10 correct (0.0%) — 37.5s |
| **Surgery** | delta_norm=0.0000 (correct 없어서 contrast pair 불가) |
| **QuickBench** | 32.0% (8/25) — PASS |
| **총 시간** | 141.7s |

### Iteration 1
| 단계 | 결과 |
|------|------|
| **Probe** | overall 13.3% (동일) |
| **Generation** | 10 problems (10 verifiable) |
| **Evaluation** | 0/10 correct (0.0%) — 35.7s |
| **Surgery** | delta_norm=0.0000 |
| **QuickBench** | 32.0% (8/25) — PASS |
| **총 시간** | 138.5s |

### 전체 결과
```
Total iterations:     2
Accepted:             0
Rejected:             2
Acceptance rate:      0.0%
Emergency rollbacks:  0
Wall time:            320.2s (5.3m)
Iter/hr:              22.5
Peak GPU memory:      13.2 GB
Best score:           0.3095
```

### 성공 기준
| 기준 | 결과 | 비고 |
|------|------|------|
| 10+ iterations 완료 | ❌ FAIL | 의도적으로 2만 실행 (smoke test) |
| 1+ surgery accepted | ❌ FAIL | 3B 모델이 문제를 다 틀려서 contrast pair 없음 |
| Score improvement | ⏭️ SKIP | 위 조건 미충족 |

## 분석

### 왜 surgery가 0건인가?

AGISTI의 surgery는 **correct vs wrong 답변의 activation 차이**를 분석해서 어디를 수정할지 결정한다. 
근데 Qwen2.5-3B가 생성된 문제 10개를 **전부 틀렸다** (0/10 correct).

→ correct 답변이 0개 = contrast pair을 만들 수 없음 = surgery proposal 불가

이건 모델이 너무 작아서가 아니라, **자기가 생성한 문제를 자기가 못 푸는** 상황.
해결: 7B 모델 사용, 또는 더 쉬운 문제 생성 로직 필요.

### 긍정적인 점 ✅

1. **전체 파이프라인이 GPU에서 에러 없이 완주**했다
2. **RTX 5090 + cu128 + bfloat16** 조합 검증 완료
3. **Peak 13.2GB / 32GB** — 7B 모델도 여유롭게 가능
4. **22.5 iter/hr** — 실전 50 iteration이면 ~2.2시간
5. **QuickBench 32.0% 일관성** — 벤치마크 시스템 안정적
6. **체크포인트 저장 정상 작동** (score=0.3095)

## 다음 단계

1. **Qwen2.5-7B로 재실행** — 더 큰 모델이면 문제를 맞출 확률 올라감
2. **iteration 수 올리기** — `--iterations 50`
3. **문제 생성 난이도 조절** — 모델이 50~70% 맞추는 난이도가 surgery에 최적
4. **HuggingFace 토큰 설정** — rate limit 경고 해소

---
*Phase 1 Smoke Test — RTX 5090 위에서 AGISTI가 숨쉬기 시작했다* 🫁

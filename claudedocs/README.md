# Moshi 소스코드 분석 문서

> 작성자: Claude (Opus 4.7) | 작성일: 2026-05-20
> 대상 커밋: `e6a55d2` (main)
> 분석 범위: PyTorch 구현 + 시스템 아키텍처 + 핵심 개념

## 문서 목록

1. **[`00-moshi-architecture-overview.md`](00-moshi-architecture-overview.md)**
   전체 모델/시스템 아키텍처 종합 분석. Mimi 코덱, RQ-Transformer (Temporal × Depth), Streaming 추상화, 시스템 토폴로지, latency budget, 생소한 개념 용어집.
   ★ 추가 섹션: § 1.3.1 (K=17 임베딩 합산의 정확한 의미), § 1.3.2 (Text 출력 경로)

2. **[`01-opus-codec.md`](01-opus-codec.md)**
   Opus 오디오 코덱이 무엇이고 Moshi 파이프라인에서 어떤 역할을 하는지. SILK/CELT 하이브리드 구조, sphn 라이브러리, Mimi와의 차이, FAQ.

3. **[`02-inner-monologue.md`](02-inner-monologue.md)**
   Moshi의 핵심 트릭인 "Inner Monologue" 심층 분석. 텍스트와 audio의 시간 정렬, padding/epad 메커니즘, 학습/추론 흐름, 인과 순서, 부수 효과(무료 STT/TTS/자막).
   ★ 추가 섹션: § 7.1 (Training asymmetry — 17 stream 입력 / 9 stream loss)

4. **[`03-training-data.md`](03-training-data.md)**
   Moshi의 6단계 학습 커리큘럼과 데이터셋 구성. Helium(text LM) → Mimi → audio pretrain → multi-stream(Fisher 등) → instruct fine-tune(합성 dialogue) → voice fine-tune(Moshiko/Moshika). 평가 지표, 본 저장소에서 확인 가능한 단서, 핵심 통찰.

## 빠른 인덱스

| 알고 싶은 것 | 보세요 |
|---|---|
| 전체 시스템이 어떻게 돌아가나? | `00` § 2.1 시스템 토폴로지 |
| 7B 모델의 정확한 구조? | `00` § 1.3 RQ-Transformer |
| K=17 codebook이 진짜로 어떻게 들어가나? | `00` § 1.3.1 (embedding sum) |
| 임베딩 테이블 17개의 정확한 매핑? | `00` § 1.3.1 |
| Text는 Mimi 거치나? (안 거침) | `00` § 1.3.2 |
| Mimi 코덱 내부? | `00` § 1.2 Mimi |
| WebSocket 위에서 뭐가 오가나? | `00` § 2.1 + `01` § 2 |
| 200 ms latency를 어떻게 달성? | `00` § 4 Latency Budget |
| 텍스트와 오디오를 같이 어떻게 생성? | `02` 전체 |
| pad(3)/epad(0) 토큰의 의미? | `02` § 3.2 + § 4.2 |
| 학습 시 어느 stream에 loss가 걸리나? | `02` § 7.1 (training asymmetry) |
| 학습 데이터를 어떻게 구성했나? | `03` 전체 |
| Fisher / Helium / Moshiko가 뭐지? | `03` § 5.3, § 2, § 7 |
| 왜 굳이 Opus를 쓰지? | `01` § 3 |
| Delay [0,0,1,1,1,...]의 의미? | `00` § 1.3 + `02` § 6 |
| 스트리밍 추상화 설계? | `00` § 2.2 StreamingModule |

## 핵심 코드 진입점 (Quick Reference)

```
모델:
  moshi/models/lm.py:49             LMModel (Moshi 7B)
  moshi/models/lm.py:556            LMGen   (추론 wrapper)
  moshi/models/compression.py:105   MimiModel (audio codec)
  moshi/models/loaders.py:90        _lm_kwargs (모든 하이퍼파라미터)
  moshi/modules/transformer.py      StreamingTransformer + RingKVCache
  moshi/modules/streaming.py:54     StreamingModule (추상 베이스)
  moshi/quantization/vq.py:170      SplitResidualVectorQuantizer
  moshi/modules/seanet.py:96        SEANetEncoder

서버:
  moshi/server.py:39                ServerState
  moshi/server.py:94                recv_loop (WebSocket 핸들러)
  moshi/server.py:154               handle_chat (WS upgrade)

클라이언트:
  moshi/client.py                   Python CLI 클라이언트
  client/src/                       TypeScript 웹 UI

Rust 백엔드:
  rust/moshi-core/src/lm.rs         LmModel (Candle 기반)
  rust/moshi-core/src/mimi.rs       Mimi
  rust/moshi-core/src/lm_generate_multistream.rs   추론 state machine
```

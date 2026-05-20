# Moshi Full-Duplex 모델 아키텍처 종합 분석

> Kyutai Moshi (arXiv:2410.00037) 소스코드 기반 모델/시스템 아키텍처 분석
> 분석 시점 커밋: `e6a55d2` (main)

---

## 0. 저장소 구조 개관 (3-Stack)

```
moshi/
├── moshi/                ← PyTorch 구현 (연구/실험용)         README.md:17
│   └── moshi/
│       ├── models/       ← MimiModel, LMModel, LMGen
│       ├── modules/      ← Transformer, SEANet, Streaming, Conv, RoPE
│       ├── quantization/ ← RVQ, SplitResidualVQ
│       ├── conditioners/ ← CFG / 텍스트 조건
│       ├── server.py     ← aiohttp WebSocket 서버
│       └── client.py     ← Python CLI 클라이언트
├── moshi_mlx/            ← MLX 구현 (Apple Silicon 로컬용)    README.md:18
├── rust/                 ← Rust 프로덕션 스택                README.md:19
│   ├── moshi-core/       ← lm, mimi, transformer, kv_cache, asr, tts
│   ├── moshi-server/     ← Axum HTTP/WS 서버
│   ├── moshi-cli/        ← TUI 클라이언트
│   └── mimi-pyo3/        ← Python 바인딩 (`rustymimi`)
└── client/               ← TypeScript/React 웹 UI            client/src/
```

핵심 아이디어: **하나의 모델 정의 (PyTorch)를 3개 백엔드 (PyTorch / MLX / Rust+Candle)로 포팅**. 모든 백엔드는 동일 WebSocket 프로토콜로 동일 웹 클라이언트와 통신.

---

## 1. 모델 아키텍처 — Mimi + Moshi (RQ-Transformer)

### 1.1 전체 데이터 흐름 (Full-Duplex)

```
                       ╔══════════════════════════════════════════╗
   USER MIC            ║       FULL-DUPLEX 양방향 동시 처리         ║          MOSHI SPEAKER
   24kHz PCM           ╚══════════════════════════════════════════╝          24kHz PCM
       │                                                                          ▲
       ▼                                                                          │
 ┌─────────────┐          ┌──────────────────────────────────┐           ┌────────────────┐
 │ Opus Decoder│          │       Mimi Audio Codec           │           │  Mimi Decoder  │
 │ (sphn)      │─PCM─────▶│   ENCODE: SEANet ↓+TR↓+RVQ       │           │ ↑ RVQ codes    │
 │ frame=1920  │ 80ms     │   Wave→latent@25Hz→latent@12.5Hz │           └────────▲───────┘
 └─────────────┘          │           ↓                      │                    │
                          │   codes [B,K=8,T] 12.5Hz, 1.1kbps│           Moshi audio codes
                          └──────────────┬───────────────────┘            [B, 8, 1] per step
                                         │
                          USER 스트림 (K_user=8 codebooks)
                                         │
                                         ▼
                          ┌──────────────────────────────────┐
                          │   Moshi LM = "RQ-Transformer"    │
                          │   ┌───────────────────────────┐  │
                          │   │  Temporal Transformer 7B  │  │ 32 layers, dim=4096
                          │   │  (frame-level, 12.5Hz)    │  │ context=3000 frames
                          │   │  Inputs (per step):       │  │
                          │   │   • text token (inner     │  │ ┐
                          │   │     monologue)            │  │ │ K=17 streams total
                          │   │   • Moshi audio (8)       │  │ │ (1 text + 8 Moshi
                          │   │   • User audio  (8)       │  │ │  + 8 user)
                          │   └─────────┬─────────────────┘  │ ┘
                          │             │  hidden_t [B, dim] │
                          │             ▼                    │
                          │   ┌───────────────────────────┐  │
                          │   │ Depth Transformer (small) │  │ 6 layers, dim=1024
                          │   │ (codebook-level, K=8)     │  │ models inter-codebook
                          │   │  Auto-regressive ACROSS   │  │ dependency at fixed t
                          │   │  the 8 codebooks per      │  │
                          │   │  TIME STEP                │  │
                          │   └─────────┬─────────────────┘  │
                          └─────────────┼────────────────────┘
                                        ▼
                          ┌──────────────────────────────────┐
                          │  Outputs per 80ms frame:         │
                          │   • text token  (1×)             │── (text WebSocket)
                          │   • Moshi audio codes (8×)       │── Mimi decode → PCM → Opus
                          └──────────────────────────────────┘
```

**핵심 사양 (loaders.py:38-119):**

| 구성요소 | 수치 | 코드 |
|---|---|---|
| 샘플레이트 | 24,000 Hz | `loaders.py:28` |
| 프레임레이트 | 12.5 Hz (80 ms/frame) | `loaders.py:29` |
| Mimi 비트레이트 | ~1.1 kbps | README.md:50 |
| RVQ 코드북 | n_q=32 (활성: 8), bins=2048 | `loaders.py:58-64` |
| Temporal Transformer | dim=4096, 32 layers, 32 heads | `loaders.py:90-99` |
| Depth Transformer | dim=1024, 6 layers, 16 heads | `loaders.py:107-110` |
| LM 입력 스트림 | K = 1(text) + 8(Moshi) + 8(user) = 17 | `loaders.py:94-95` |
| Context window | 3000 frames ≈ 240 s | `loaders.py:102` |
| Position embedding | RoPE | `loaders.py:106` |
| Norm | rms_norm_f32 | `loaders.py:105` |
| FFN gating | SiLU (SwiGLU) | `loaders.py:104` |

> Opus Decoder에 대한 설명은 [`01-opus-codec.md`](01-opus-codec.md) 참조.
> Inner Monologue에 대한 심층 분석은 [`02-inner-monologue.md`](02-inner-monologue.md) 참조.

---

### 1.2 Mimi — Neural Audio Codec

```
                  Mimi: STREAMING NEURAL AUDIO CODEC                 README.md:47-67
                  =========================================
                  24kHz audio  ⇒  12.5Hz, 1.1 kbps codes
                  Latency: 80ms (one frame)

   PCM 24kHz                                                                PCM 24kHz
   [B, 1, T_wave]                                                          [B, 1, T_wave]
        │                                                                       ▲
        ▼                                                                       │
   ┌────────────────────┐                                          ┌────────────────────┐
   │ SEANetEncoder      │   seanet.py:96     hop_length=8·6·5·4    │ SEANetDecoder      │
   │ • Causal Conv1d    │                    = 960 (→ 24kHz/960    │ • TransposedConv1d │
   │ • 4× downsample    │                       =25Hz)             │ • 4× upsample      │
   │   ratios=[8,6,5,4] │                                          │   ratios=[8,6,5,4] │
   │ • ResNet blocks    │                                          │ • ResNet blocks    │
   │ • ELU              │                                          │ • ELU              │
   └─────────┬──────────┘                                          └──────────▲─────────┘
             ▼  [B, 512, T_lat@25Hz]                                          │
   ┌────────────────────┐                                          ┌────────────────────┐
   │ Encoder Transformer│   loaders.py:65-80                       │ Decoder Transformer│
   │ 8 layers, dim=512  │   compression.py:312-313                 │ 8 layers, dim=512  │
   │ causal, RoPE       │   ProjectedTransformer                   │ causal, RoPE       │
   └─────────┬──────────┘                                          └──────────▲─────────┘
             ▼                                                                │
   ┌────────────────────┐                                          ┌────────────────────┐
   │ Downsample (conv)  │   compression.py:202   25Hz → 12.5Hz    │ Upsample (conv)    │
   │ stride=2           │   _to_framerate()                        │ stride=2           │
   └─────────┬──────────┘                                          └──────────▲─────────┘
             ▼  [B, 512, T@12.5Hz]                                            │
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │   SplitResidualVectorQuantizer    quantization/vq.py:170                     │
   │   ┌────────────────┐    ┌────────────────────────────────────────┐           │
   │   │ rvq_first      │    │  rvq_rest                              │           │
   │   │ n_q=1          │    │  n_q=7   (codebook_offset=1)           │           │
   │   │ "SEMANTIC"     │ +  │  "ACOUSTIC"                            │           │
   │   │ WavLM distill  │    │  multiple residual codebooks           │           │
   │   └────────────────┘    └────────────────────────────────────────┘           │
   │   codes [B, K=8, T]   bins=2048,   12.5Hz × log2(2048) × 8 ≈ 1.1 kbps        │
   └──────────────────────────────────────────────────────────────────────────────┘
```

**핵심 기법 (vq.py:170-322 + README.md:47-61):**
- **Split RVQ**: 첫 코드북 1개는 **WavLM(self-supervised)에 distillation** ⇒ 의미(semantic) 정보, 나머지 7개 잔차는 **음향(acoustic)** 디테일.
- **Streaming**: 모든 Conv가 **causal padding** (StreamingConv1d, `modules/conv.py`).
- **Adversarial-only loss** (no reconstruction L1/L2) → 1.1 kbps의 극저 비트레이트에도 주관적 품질 우수 (README:60).
- **Distillation**: 첫 코드북은 자기 자신을 예측하면서 동시에 WavLM 임베딩과 비슷해지도록 학습되어 semantic + acoustic이 한 모델에 통합됨.

---

#### 1.2.1 ⚠️ "Semantic codebook"은 텍스트 토큰이 아니다

`semantic`이라는 단어가 두 군데에 쓰이는데, 둘은 완전히 다른 것:

| 구분 | Mimi semantic codebook (M0, U0) | Text token (Inner Monologue, T) |
|------|-------------------------------|----------------------------------|
| 종류 | AUDIO 코드 (RVQ 정수) | TEXT 토큰 (SentencePiece subword) |
| Vocab 크기 | 2048 | 32000 |
| 출처 | `Mimi.encode(waveform)` | `LM.text_linear` 직접 sample |
| "semantic"인 이유 | WavLM SSL 임베딩과 cosine 정렬 (distillation) | 본질적으로 텍스트라서 의미적 |
| 무엇을 담나 | 음소·운율의 의미적 부분 (audio domain) | 실제 단어 ("hello", "▁world") |
| Frame당 개수 | 1 per side (Moshi M0 + User U0 = 2개) | 1 (Moshi 측만) |
| 출력 경로 | Depth → Mimi.decode → PCM | text_linear → WebSocket `\x02` |
| 코드 | `quantization/vq.py:170` (rvq_first) | `lm.py:141` (text_linear) |
| 다이어그램 위치 | Mimi 코덱 내부 | LM 외부 — Mimi 안 거침 |

**둘은 보완 관계로 공존**:
- T = "무엇을" (단어)
- M0 = "어떻게 시작" (음소·억양의 의미적 부분)
- M1..M7 = "정확한 음향 디테일" (자연스러운 음질)

Inner Monologue 인과 사슬: `text_t → M0_t → M1_t → ... → M7_t` (Depth Transformer 내부, lm.py:818).

---

#### 1.2.2 Mimi 코드 shape · bins · bitrate 해석

`codes [B, K=8, T]`, `bins=2048`, `12.5Hz × log₂(2048) × 8 ≈ 1.1 kbps` 의 의미:

**Shape 분해:**

```
   codes.shape = [B, K=8, T]
                  ↑   ↑    ↑
                  │   │    └─ T: 시간축 frame 수 (12.5 frame/sec)
                  │   └────── K: codebook 개수 (RVQ depth = 8)
                  └────────── B: batch size

   각 원소 codes[b, k, t] ∈ {0, 1, ..., 2047}    ← 정수 ID
                            └────── bins = 2048: 각 codebook의 vocabulary size
```

**Bitrate 계산:**

```
   bits per frame = K codebooks × log₂(bins)
                  = 8 × log₂(2048)
                  = 8 × 11 bits
                  = 88 bits

   bits per second = frame_rate × bits_per_frame
                   = 12.5 Hz × 88 bits
                   = 1100 bits/sec
                   = 1.1 kbps
```

**입출력 단위 통일 비교 (1초 분량 기준):**

```
   ┌─────────────────────┬──────────────┬───────────┬──────────────┐
   │ 표현 단위            │ 1초당 개수    │ 단위 크기 │ 1초당 bit량   │
   ├─────────────────────┼──────────────┼───────────┼──────────────┤
   │ PCM sample (raw)    │  24,000      │ 16 bits   │  384,000 bit │
   │ Opus packet (net)   │   ~50        │ varies    │   ~24,000 bit│
   │ Mimi frame          │     12.5     │ 88 bits   │    1,100 bit │
   │   = 8 codebooks     │  (12.5×8=100)│ 11 bits   │    1,100 bit │
   │ Text token (T)      │     12.5     │ ~15 bits  │    ~187 bit  │
   │   (raw log₂(32k))   │              │           │              │
   │ Text 실효 (단어)     │   ~3 (word) │ ~15 bits  │    ~45 bit   │
   └─────────────────────┴──────────────┴───────────┴──────────────┘
```

**Frame당 한 묶음 (80 ms = 1 Mimi frame):**

```
   1 Mimi frame (80 ms) 안에 들어 있는 것:
   ┌──────────────────────────────────────────────────────────────────┐
   │  • 원본 PCM   : 1920 samples × 16 bits = 30,720 bits             │
   │  • Mimi codes : K=8개 정수 (각 11 bits) = 88 bits      [349× 압축]│
   │  • Text token : 1개 정수 (~15 bits)    = ~15 bits                 │
   └──────────────────────────────────────────────────────────────────┘
```

**핵심 비유:**
- **PCM sample** = 원본 마이크 수치 (1초에 24,000개)
- **Mimi frame** = 80ms 단위로 묶어 8개 정수로 압축한 "오디오 토큰 묶음"
- **codebook 1개** = 그 묶음 안의 한 "축"; 첫 축(M0)은 semantic 정보, 나머지 7축은 acoustic 디테일
- **text token** = 그 80ms 동안 발화 중인 "단어 조각"

→ Mimi codes는 텍스트 토큰과 **시간축은 같은 12.5 Hz지만 vocab과 의미가 완전히 다른** 별개 stream.

---

#### 1.2.3 왜 `K=8`, `bins=2048` (= 11 bits)인가? — 설계 근거

세 가지 제약의 교집합:

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  ① 음성 품질을 위한 최소 비트레이트                                  │
   │     • 사람이 알아들을 만한 음성: 통상 1~4 kbps 필요                  │
   │     • neural codec + adversarial loss 덕에 1.1 kbps까지 내림        │
   │     • 그 이하면 metallic/robotic                                    │
   │                                                                      │
   │  ② Depth Transformer의 autoregressive 비용 (latency)                 │
   │     • 매 frame마다 Depth가 K번 AR step                              │
   │     • K=8이면 1 frame당 1 Temporal + 8 Depth = 9 forward            │
   │     • K=16이면 1 + 16 = 17 forward → 2배 느림 (200ms 깨짐)           │
   │     • K=4이면 빠르지만 음질 하락                                    │
   │                                                                      │
   │  ③ Codebook 학습 가능성                                              │
   │     • bins ↑ → 표현력 ↑ 그러나 rare entry로 인해 학습 불안정          │
   │     • bins ↓ → 표현력 ↓ 음향 디테일 손실                            │
   │     • 1024~4096 사이가 sweet spot (EnCodec/SoundStream 선례)        │
   │                                                                      │
   │  교집합: K=8, bins=2048                                              │
   │     → 8 × log₂(2048) = 88 bits/frame                                │
   │     → 88 × 12.5Hz = 1.1 kbps                                        │
   └────────────────────────────────────────────────────────────────────┘
```

**Frame당 비트 예산 88 bits의 의미:**

```
   88 bits/frame을 어떻게 8개로 쪼개는가:
   ┌────────────────────────────────────────────────────────────────┐
   │  codebook 0 (semantic, WavLM-distilled): 11 bits                │
   │     "이 80ms에 무슨 음소가 시작됐나 / 단어 식별성"               │
   │  codebook 1: 11 bits                                            │
   │     "위 정보 빼고 남은 잔차의 첫 번째 양자화"                    │
   │  codebook 2: 11 bits                                            │
   │     "여기까지 빼고 남은 잔차의 두 번째 양자화"                   │
   │     ⋮                                                            │
   │  codebook 7: 11 bits                                            │
   │     "마지막 미세 acoustic texture"                              │
   │                                                                  │
   │  → 깊이가 깊을수록 더 미세한 정보 (RVQ의 본질)                   │
   │  → 앞쪽 codebook이 가장 중요 → semantic을 0번에 둠               │
   └────────────────────────────────────────────────────────────────┘
```

**bins=2048이 power-of-2인 이유:**
- log₂(2048) = 11 **딱 떨어짐** → bit 계산 깔끔
- vocab embedding size, hashing, bit packing 모두 효율적
- 2049 = 2048 + 1 (special `<initial>` 토큰 자리, `lm.py:136`)

**왜 `n_q=32` 학습, `num_codebooks=8` 추론?** (`loaders.py:60`, `compression.py:255-260`)
- RVQ는 학습 시 **변동 깊이**로 학습됨 (1 ~ 32 무작위) — quantizer dropout
- 추론 시에는 8개만 활성화 (set_num_codebooks) → 1.1 kbps에 맞춤
- 필요시 16, 32까지 늘려 고품질 모드도 가능 (대신 latency 증가)

**대안 시뮬레이션:**

| K | bins | bits/frame | bitrate | Depth steps/frame | 평가 |
|---|------|-----------|---------|------------------|------|
| 4 | 2048 | 44 | 0.55 kbps | 4 | 빠름 / 음질 ↓↓ |
| **8** | **2048** | **88** | **1.1 kbps** | **8** | **✅ 채택** |
| 8 | 4096 | 96 | 1.2 kbps | 8 | 비슷, 학습 어려움 |
| 16 | 2048 | 176 | 2.2 kbps | 16 | 음질 ↑ / latency 2× |
| 32 | 2048 | 352 | 4.4 kbps | 32 | 학습용 풀 RVQ |

→ K=8, bins=2048은 **품질 / 지연 / 학습 가능성의 균형점**.

---

#### 1.2.4 ⚠️ RVQ ≠ Mimi — 위치와 학습 방법

**RVQ는 Mimi의 한 부품이지 Mimi 전체가 아닙니다.**

```
   ┌────────────────────────────────────────────────────────────────────┐
   │              Mimi 전체 파이프라인 (compression.py:105)               │
   │                                                                      │
   │  PCM 24kHz                                          PCM 24kHz        │
   │      ↓                                                  ↑           │
   │  ┌──────────────┐                              ┌──────────────┐     │
   │  │ SEANetEncoder│  ① 신경망 (conv stack)        │ SEANetDecoder│     │
   │  └──────┬───────┘                              └──────▲───────┘     │
   │         ↓ 25 Hz latent                                │             │
   │  ┌──────────────┐                              ┌──────────────┐     │
   │  │ Transformer  │  ② 신경망 (8-layer)           │ Transformer  │     │
   │  │   encoder    │                              │   decoder    │     │
   │  └──────┬───────┘                              └──────▲───────┘     │
   │         ↓                                             │             │
   │  ┌──────────────┐                              ┌──────────────┐     │
   │  │ Downsample   │  ③ 학습된 stride conv         │ Upsample     │     │
   │  │ 25→12.5 Hz   │                              │ 12.5→25 Hz   │     │
   │  └──────┬───────┘                              └──────▲───────┘     │
   │         ↓ 12.5 Hz latent                              │             │
   │  ╔══════════════╗                              ╔══════════════╗     │
   │  ║ ★ RVQ ★      ║  ④ Vector Quantizer          ║ RVQ.decode   ║     │
   │  ║ (이 부분만)   ║   - 연속 → 정수 코드          ║ 정수 → 연속  ║     │
   │  ║ vq.py:170    ║   - core_vq.py:437           ║              ║     │
   │  ╚══════╤═══════╝                              ╚══════▲═══════╝     │
   │         ↓ 정수 codes [B, 8, T]                        │             │
   │         └─ Moshi LM 또는 디스크 저장 ────────────────┘             │
   └────────────────────────────────────────────────────────────────────┘

   = Mimi = SEANet 인코더 + Transformer + RVQ + Transformer + SEANet 디코더
   = RVQ = 그 안의 양자화 단계 1개
```

---

#### 1.2.5 codebook 0~7의 실체와 RVQ 학습

**codebook 1개의 실체** (`core_vq.py:105-160` `EuclideanCodebook`):

```
   codebook 0이 가진 것:
   ┌───────────────────────────────────────────────────────────────┐
   │  embedding ∈ ℝ^(2048 × 256)    ← 2048개의 centroid 벡터        │
   │                                                                 │
   │  centroid_0    = [0.12, -0.34, 0.56, ..., 0.78]  (256-dim)     │
   │  centroid_1    = [-0.45, 0.78, 0.23, ..., 0.11]                │
   │   ⋮                                                             │
   │  centroid_2047 = [0.92, 0.15, -0.67, ..., 0.42]                │
   │                                                                 │
   │  + buffers (학습 중에만 사용):                                  │
   │    - cluster_usage  ∈ ℝ^2048   ← 각 centroid 사용 빈도 EMA      │
   │    - embedding_sum  ∈ ℝ^(2048×256) ← 할당된 벡터 누적합 EMA    │
   │                                                                 │
   │  Frame당 출력: 정수 1개 ∈ {0, 1, ..., 2047}                     │
   │     "이 frame의 잔차 벡터와 가장 가까운 centroid의 index"        │
   │                                                                 │
   │  메모리:                                                        │
   │    2048 centroids × 256 dim × 4 bytes(fp32) ≈ 2 MB / codebook  │
   │    × 8 codebooks ≈ 16 MB (Mimi 전체 codebook 메모리)            │
   └───────────────────────────────────────────────────────────────┘

   💡 비유: codebook은 "2048개 단어가 적힌 사전",
            출력 11 bits는 "사전의 페이지 번호".
            출력 = 사전 그 자체가 아니라 그 안의 한 entry를 가리키는 인덱스.
```

**Frame당 codebook 0~7이 만드는 것** (`core_vq.py:469-484` `ResidualVectorQuantization.forward`):

```
   ┌────────────────────────────────────────────────────────────────┐
   │ 한 frame의 잔차 양자화 사슬 (입력 x ∈ ℝ²⁵⁶):                     │
   │                                                                  │
   │  step 0:                                                         │
   │    잔차 = x                                                      │
   │    idx_0 = argmin_i ‖잔차 − codebook0[i]‖   ← code 0 (11 bits)  │
   │    q_0 = codebook0[idx_0]                                       │
   │    잔차 ← 잔차 − q_0   (다음 step을 위한 새 잔차)               │
   │                                                                  │
   │  step 1:                                                         │
   │    idx_1 = argmin_i ‖잔차 − codebook1[i]‖   ← code 1 (11 bits)  │
   │    q_1 = codebook1[idx_1]                                       │
   │    잔차 ← 잔차 − q_1                                            │
   │                                                                  │
   │   ⋮                                                              │
   │                                                                  │
   │  step 7:                                                         │
   │    idx_7 = argmin_i ‖잔차 − codebook7[i]‖   ← code 7 (11 bits)  │
   │                                                                  │
   │  최종 복원: x̂ = q_0 + q_1 + ... + q_7                            │
   │  최종 코드: [idx_0, idx_1, ..., idx_7]  ← 8개 정수 (88 bits)     │
   └────────────────────────────────────────────────────────────────┘
```

**RVQ 학습 메커니즘 — 표준 backprop이 아님:**

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  3중 학습 신호                                                     │
   │                                                                    │
   │  ① 인코더 학습 — Straight-Through Estimator (STE)                  │
   │     core_vq.py:426                                                 │
   │     quantized = x + (quantized - x).detach()                       │
   │                  └─── forward는 양자화된 값                          │
   │                       backward는 x로 그대로 흘려보냄                  │
   │     → argmin은 미분 불가하지만 STE로 gradient 통과                  │
   │     → 인코더(SEANet+Transformer)는 표준 backprop로 학습              │
   │                                                                    │
   │  ② Codebook 학습 — EMA (Exponential Moving Average)                │
   │     core_vq.py:322-335   ← Gradient descent 아님!                   │
   │     매 step:                                                       │
   │       - 각 centroid에 어떤 벡터들이 할당됐는지 집계                  │
   │       - embedding_sum ← decay × embedding_sum + (1-decay) × 새합     │
   │       - cluster_usage ← decay × cluster_usage + (1-decay) × 새카운트  │
   │       - 실제 centroid = embedding_sum / cluster_usage                │
   │     → 본질적으로 "online k-means" — 클러스터 중심을 점진 이동         │
   │     → 학습률 = (1 - 0.99) = 0.01 (decay=0.99, vq.py:54)              │
   │                                                                    │
   │  ③ Commitment Loss — 인코더가 centroid 근처를 출력하게 강제          │
   │     core_vq.py:427                                                 │
   │     loss = F.mse_loss(x, quantized.detach())                       │
   │     → 인코더 출력이 가까운 centroid에 "약속(commit)"되게 함          │
   │     → 이 loss는 인코더 weight만 업데이트 (centroid는 detach됨)        │
   └──────────────────────────────────────────────────────────────────┘
```

**부가 학습 메커니즘:**

| 메커니즘 | 코드 | 역할 |
|---------|------|------|
| **K-means 초기화** | `core_vq.py:77-97, 196-222` | 첫 배치에서 centroid 초기값 결정 (random보다 빠른 수렴) |
| **Dead code 부활** | `core_vq.py:229-260` | 5 step마다 거의 사용 안 되는 centroid를 현재 batch의 랜덤 벡터로 교체 → "codebook collapse" 방지 |
| **WavLM Distillation** | (학습 코드 비공개) | codebook 0에 한해 출력이 WavLM 임베딩과 cosine 유사하도록 추가 loss |
| **Adversarial loss** | (학습 코드 비공개) | Mimi.decode 출력 PCM이 진짜 같도록 discriminator로 학습 |

**Mimi 전체의 학습 손실 (논문 §4):**

```
   L_total = L_adversarial          (MS-STFT discriminator)
           + L_feature_matching     (discriminator 중간 feature 매칭)
           + L_commitment           (RVQ encoder ↔ centroid)
           + L_distill              (codebook 0 ↔ WavLM)
           # 명시적 L_recon (L1/L2) 없음!
```

**RVQ가 Mimi 전체와 함께 end-to-end로 학습되는 핵심:**
- 인코더/디코더는 표준 gradient로 학습
- RVQ centroid는 EMA로 학습 (gradient 아님)
- 둘이 STE로 연결되어 한 forward/backward에 같이 업데이트
- Mimi 전체는 adversarial로 최종 음질 추구

**STE의 시각화 — 어떻게 미분 불가한 argmin을 통과시키나:**

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  Forward 경로 (실제 양자화 수행):                                  │
   │                                                                   │
   │      ┌─────┐   argmin     ┌─────────┐                            │
   │      │  x  │ ────────────▶│   q     │ ← centroid 값 사용          │
   │      └─────┘  (lookup)    └─────────┘                            │
   │                                                                   │
   │  Backward 경로 (gradient 통과):                                    │
   │                                                                   │
   │      ┌─────┐  identity     ┌─────────┐                           │
   │      │∂L/∂x│ ◀────────────│ ∂L/∂q   │ ← argmin이 없었던 것처럼    │
   │      └─────┘   (그대로)    └─────────┘                           │
   │                                                                   │
   │  코드 한 줄로:                                                     │
   │      quantized = x + (quantized - x).detach()                    │
   │                  └──────────────────────┘                         │
   │      forward 값: quantized                                        │
   │      backward 시 미분: ∂(x + 0)/∂x = 1   (그대로 통과)             │
   └─────────────────────────────────────────────────────────────────┘
```

**Mimi 한 step의 학습 흐름 — 종합 다이어그램:**

```
   ╔══════════════════════════════════════════════════════════════════════╗
   ║              Mimi 한 step의 학습 흐름                                  ║
   ╠══════════════════════════════════════════════════════════════════════╣
   ║                                                                        ║
   ║  audio batch (24kHz PCM)                                              ║
   ║      │                                                                  ║
   ║      ▼                                                                  ║
   ║  SEANet enc + Transformer  ←──── ⑤ gradient (STE 통과) ──┐              ║
   ║      │                                                    │              ║
   ║      ▼ x ∈ ℝ²⁵⁶ (12.5Hz latent)                          │              ║
   ║  ┌──────────────────────────────────────────────────┐    │              ║
   ║  │ RVQ (codebooks 0~7)                              │    │              ║
   ║  │   forward: argmin lookup → 8 정수                 │    │              ║
   ║  │   STE:    q ← x + (q − x).detach()              │────┘              ║
   ║  │   EMA:    centroid 위치 update ① (gradient 아님) │                    ║
   ║  │   Commit: L_commit = ‖x − sg(q)‖² → 인코더로 ② grad                  ║
   ║  │   Distill: codebook 0 ↔ WavLM ③ (gradient)                          ║
   ║  └──────────┬───────────────────────────────────────┘                  ║
   ║             ▼ q (quantized continuous vector)                           ║
   ║  Transformer dec + SEANet dec                                           ║
   ║             │                                                            ║
   ║             ▼ audio_hat (24kHz PCM reconstruction)                       ║
   ║  Discriminator (MS-STFT)                                                ║
   ║             │                                                            ║
   ║             ▼ L_adv, L_fm → 전체 신경망에 gradient ④                    ║
   ║                                                                          ║
   ║  최종 업데이트:                                                          ║
   ║    • 인코더/디코더/transformer: ②+③+④의 gradient 합으로 backprop        ║
   ║    • codebook centroid:        ①의 EMA만 (gradient와 무관, k-means식)  ║
   ║    • dead code:                5 step마다 점검 후 부활                   ║
   ║                                                                          ║
   ║  명시적으로 없는 것: L1/L2 reconstruction loss                           ║
   ║                     → 적대적 + perceptual feature matching이 그 역할     ║
   ╚══════════════════════════════════════════════════════════════════════╝
```

**Mimi.encode 내부 호출 순서 (`compression.py:338-388`):**

```python
def encode(self, audio):                          # audio: [B, 1, T_wave] @24kHz
    emb = self.encoder(audio)                     # SEANet enc → [B,512,T@25Hz]
    emb = self.encoder_transformer(emb)           # Transformer 정제
    emb = self._to_framerate(emb)                 # 25Hz → 12.5Hz downsample
    codes = self.quantizer.encode(emb)            # ★ RVQ ★ → [B,8,T@12.5Hz] 정수
    return codes
```

→ Mimi는 RVQ 위에 신경망 4겹(SEANet enc, Transformer enc, downsample, +역방향)을 덧씌운 것.
→ RVQ는 그 신경망들에 의해 만들어진 256-d latent를 정수 코드로 양자화하는 "병목 단계".

---

### 1.3 Moshi LM — RQ-Transformer (Temporal × Depth)

**RQ-Transformer의 의미**: "Residual Quantizer + Transformer". 잔차 양자화로 만든 K개의 코드북을 **두 축**으로 모델링. 시간축은 큰 Transformer, 코드북축은 작은 Transformer.

```
              RQ-TRANSFORMER (Moshi LM)             lm.py:49 (LMModel)
              ====================================

  Input frame_t (in streaming, K=17 codebooks for previous step t-1):
   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   │T │M0│M1│M2│M3│M4│M5│M6│M7│U0│U1│U2│U3│U4│U5│U6│U7│   K = 1 + 8 + 8 = num_codebooks
   └─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┴─┬┘   audio_offset=1, dep_q=8, n_q=16
     │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
   ┌─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴────────┐  lm.py:135-139
   │  ScaledEmbedding × 17  (text_emb + emb[i])                │  forward_text() lm.py:379-408
   │  Σ → input [B, T, dim=4096]                               │
   └──────────────────────┬────────────────────────────────────┘
                          ▼
   ┌───────────────────────────────────────────────────────────┐
   │  TEMPORAL TRANSFORMER  (StreamingTransformer)             │  loaders.py:97-106
   │  32 layers × 32 heads × dim=4096, hidden_scale=4.125      │  total ≈ 7B params
   │  • RoPE positional embedding   modules/rope.py            │
   │  • SwiGLU FFN (gating="silu")  modules/gating.py          │
   │  • RMSNorm-f32 (numerical stab)                           │
   │  • Causal mask, context=3000   modules/transformer.py     │
   │  • RingKVCache  (streaming!)   transformer.py:196-...     │
   └──────────┬────────────────────────────────┬───────────────┘
   transformer_out [B, T, 4096]                ▼
              │                            ┌──────────────────────────┐
              ▼                            │  text_linear             │  lm.py:141
   ┌────────────────────────┐              │  → text_logits [B,1,T,V] │  V_text=32000
   │ depformer_in (per-cb)  │              └──────────────────────────┘
   │ 8× nn.Linear           │              sample text token → t_text
   │ 4096 → 1024            │              lm.py:736-745
   │ depformer_multi_linear │
   └────────────┬───────────┘                                                lm.py:809-850
                ▼                                                            (depformer_step)
   ╔════════════════════════════════════════════════════════════════════╗
   ║   DEPTH TRANSFORMER — autoregressive ACROSS K codebooks (dep_q=8)  ║   loaders.py:107-118
   ║   d_model=1024, 6 layers, 16 heads, weights_per_step=True           ║   lm.py:204-218
   ║                                                                      ║
   ║   step 0: input = text_token_t                  ─► audio code A0    ║
   ║   step 1: input = A0                            ─► audio code A1    ║
   ║   step 2: input = A1                            ─► audio code A2    ║
   ║    ...                                                              ║
   ║   step 7: input = A6                            ─► audio code A7    ║
   ║                                                                      ║
   ║   each step adds transformer_out (residual signal from Temporal)    ║
   ║   detached streaming: own KV cache, reset every frame               ║   lm.py:218
   ╚═══════════════════════════════════╤════════════════════════════════╝
                                       ▼
                               Moshi audio codes
                                 [B, 8, 1]      (delayed write back into cache)
```

**RQ-Transformer 작동 원리 (lm.py:668-783, `_step`):**

```
Frame t의 한 스텝:
  1) User 오디오 8 코드를 캐시에 기록 (lm.py:693-696)
  2) 캐시에서 모든 17 채널의 t-1 토큰을 gather → embedding sum (forward_text)
  3) Temporal Transformer 한 스텝 → transformer_out, text_logits  (lm.py:726)
  4) text_logits에서 텍스트 토큰 샘플링                            (lm.py:736)
  5) Depth Transformer가 8단계 autoregressive로 audio codes 8개 샘플 (lm.py:823-841)
  6) text + audio 토큰을 캐시에 기록 (delayed positions)            (lm.py:762-772)
```

**Delayed Streams (lm_utils.py:9-38):**
서로 다른 코드북에 **양의 오프셋 지연**을 두어 정렬을 변경 — semantic(첫 코드북)을 먼저 예측한 후 그 정보를 활용해 acoustic 코드북 예측 ⇒ 품질↑.
설정: `delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]` (loaders.py:118).
- text와 첫 Moshi 코드북: delay 0
- 나머지 Moshi acoustic: delay 1 frame
- user 첫 코드북: delay 0, 나머지 user: delay 1

**Inner Monologue** (README.md:36-37): Moshi는 **자기 자신의 발화 텍스트를 audio 코드와 함께 동시 생성**. 텍스트가 곧 generation의 "골격" 역할 → 일관성/품질 크게 향상.
→ [`02-inner-monologue.md`](02-inner-monologue.md)에서 상세 다이어그램.

---

### 1.3.1 ⚠️ "K=17 codebook이 들어온다"의 정확한 의미

다이어그램의 17개 입력 박스가 시퀀스 길이 17이 아니라는 점을 명확히 합니다.

```
   ╔══════════════════════════════════════════════════════════════════════╗
   ║  한 시점 t의 입력 변환 흐름                                            ║
   ╠══════════════════════════════════════════════════════════════════════╣
   ║                                                                        ║
   ║  Step 1) 17개 정수 토큰 ID  (codebook은 정수 ID. 벡터 아님!)           ║
   ║   T(t-1)=17234, M0=412, M1=893, ..., M7=1024, U0=773, ..., U7=298     ║
   ║                                                                        ║
   ║  Step 2) 17개 별개 Embedding Table에서 lookup    lm.py:135-139         ║
   ║   text_emb[17234]   → v_T   ∈ ℝ⁴⁰⁹⁶                                    ║
   ║   emb[0][412]       → v_M0  ∈ ℝ⁴⁰⁹⁶                                    ║
   ║   emb[1][893]       → v_M1  ∈ ℝ⁴⁰⁹⁶                                    ║
   ║       ⋮                                                                ║
   ║   emb[15][298]      → v_U7  ∈ ℝ⁴⁰⁹⁶                                    ║
   ║                                                                        ║
   ║  Step 3) ⚠️ ELEMENT-WISE SUM (17 → 1)            lm.py:394, 397        ║
   ║   h_in = v_T + v_M0 + v_M1 + ... + v_U7  ∈ ℝ⁴⁰⁹⁶                        ║
   ║                                                                        ║
   ║  Step 4) Temporal Transformer 입력 shape:  [B, S=time, 4096]           ║
   ║          K=17은 attention 차원이 아니라 step 3에서 이미 합산되어 사라짐 ║
   ╚══════════════════════════════════════════════════════════════════════╝
```

**17개 별개 임베딩 테이블의 정확한 매핑:**

| #  | 변수 | 담당 codebook | vocab | dim |
|----|------|---------------|-------|-----|
| 1  | `self.text_emb` | TEXT (Inner Monologue) | 32001 | 4096 |
| 2  | `self.emb[0]` | Moshi M0 (semantic) | 2049 | 4096 |
| 3-9| `self.emb[1..7]` | Moshi M1..M7 (acoustic) | 2049 | 4096 |
| 10 | `self.emb[8]` | User U0 (semantic) | 2049 | 4096 |
| 11-17| `self.emb[9..15]` | User U1..U7 (acoustic) | 2049 | 4096 |

→ **codebook 자리마다 자신만의 임베딩 사전**을 가짐. 같은 정수 ID(예: 412)라도 자리가 다르면(semantic vs acoustic, Moshi vs User) 완전히 다른 벡터로 매핑됨.

**총 임베딩 파라미터:**
- text_emb: 32001 × 4096 ≈ **131 M**
- emb[0..15]: 16 × 2049 × 4096 ≈ **134 M**
- 합계 ≈ **265 M** (전체 7B 모델 중 임베딩만 4% 수준)

**왜 17개 벡터를 "합산"하는가? — 대안과 비교:**

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 대안 1) K개를 시퀀스 차원으로 펼침 ("flatten")                       │
   │   - 시퀀스 길이 × 17 → 자기회귀 step도 17배 → 너무 느림              │
   │   - 200ms latency 불가능                                            │
   │                                                                       │
   │ 대안 2) K개를 concat → Linear(dim × K → dim)                         │
   │   - 입력 dim이 17 × dim_emb로 폭발 → 메모리 부담                    │
   │   - 수학적으로는 합산과 동등하지만 비효율                            │
   │                                                                       │
   │ ✅ 채택) 각자 별개 임베딩 후 element-wise SUM                        │
   │   - 시퀀스 길이 변화 없음 (속도 영향 0)                              │
   │   - 임베딩 테이블 K개로 표현력은 K배                                 │
   │   - "각 codebook이 hidden space에 자신의 contribution을 더한다"      │
   │   - AudioCraft/MusicGen (Copet et al. 2023) 표준 트릭               │
   └─────────────────────────────────────────────────────────────────────┘
```

**왜 codebook 자리마다 별개 테이블인가? — 대안과 비교:**

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 대안 A) 모든 audio codebook이 1개 테이블 공유                        │
   │   - 토큰 ID는 같은 [0, 2048] 범위라 lookup은 됨                     │
   │   - ❌ M0(semantic) 토큰 #412와 M7(acoustic) #412는 의미 완전 다름   │
   │   - ❌ Moshi M0와 User U0도 의미 다름 (화자 분리 신호 손실)          │
   │                                                                       │
   │ 대안 B) Moshi 1개 + User 1개 + codebook-position embedding 추가      │
   │   - 17개 → 3개로 축소 가능                                          │
   │   - △ 학습 가능하지만 표현력 부족                                    │
   │                                                                       │
   │ ✅ 채택) codebook 자리마다 별개 테이블 (text 1 + audio 16)           │
   │   - "이 자리에 오는 토큰의 의미"가 자리마다 다르다는 점을 명시       │
   │   - semantic vs acoustic 분리, Moshi vs User 분리                   │
   │   - 파라미터 ~134M로 비싸지만 표현력 ↑                              │
   └─────────────────────────────────────────────────────────────────────┘
```

---

### 1.3.2 ⚠️ Text의 출력 경로 — Mimi를 거치지 않음

다이어그램에서 텍스트가 어디로 나가는지 명확히 합니다.

```
   ┌───────────────────────────────────────────────────────────────────┐
   │             Temporal Transformer output h_t                         │
   └─────────┬─────────────────────────────────────┬───────────────────┘
             ▼                                     ▼
   ┌──────────────────┐                  ┌──────────────────────┐
   │ self.text_linear │                  │ depformer_in (8×)    │
   │ 4096 → 32000     │   lm.py:141      │ 4096 → 1024          │  lm.py:179-181
   └────────┬─────────┘                  └──────────┬───────────┘
            ▼                                       ▼
   text_logits [B,1,T,32000]                 Depth Transformer
            │                                       │ (8 step AR)
            ▼                                       ▼
   sample text_t  lm.py:736                  Moshi audio codes M0..M7
            │                                       │
       ┌────┴────┐                                  ▼
       ▼         ▼                          Mimi.decode → PCM → Opus
   외부 송신   Depth seed                          │
   server.py:  lm.py:818                          ▼
   86-92       (텍스트가 Depth의                ws.send_bytes(b"\x01"+...)
       │       첫 입력으로 재사용)
       ▼
   ws.send_bytes(b"\x02" + utf8_text)
   → 브라우저 자막
```

**핵심 비대칭:**
- **TEXT 출력 경로**: Temporal → `text_linear` → 자체 head로 즉시 token 산출 → WebSocket `\x02`
- **AUDIO 출력 경로**: Temporal → Depth → 8 codebook codes → `Mimi.decode` → PCM → Opus → WebSocket `\x01`
- **Text는 절대 Mimi를 거치지 않음**. Mimi는 오디오 토큰만 다룸.
- 같은 text token이 **(1) 외부 자막으로 송신** + **(2) Depth Transformer의 첫 step 입력**으로 동시 사용 → Inner Monologue 메커니즘의 핵심.

---

## 2. 시스템 아키텍처 — Streaming Inference Pipeline

### 2.1 전체 시스템 토폴로지

```
       ┌─────────────────────┐       WebSocket /api/chat (binary)        ┌─────────────────────────┐
       │  Browser / Web UI   │◀────────────────────────────────────────▶│  aiohttp Server         │
       │  client/src/...     │       1-byte tag protocol                │  moshi/server.py        │
       │                     │                                          │                          │
       │ MicWorklet ─▶ Opus  │  ─── b"\x01" + opus_bytes (audio in) ──▶│  recv_loop              │
       │                     │                                          │   server.py:94-150       │
       │ Audio decoder       │  ◀── b"\x01" + opus_bytes (audio out) ──│   handle_chat            │
       │ Text renderer       │  ◀── b"\x02" + utf8_text   (text out) ──│   server.py:154-169      │
       │                     │  ◀── b"\x00"  (handshake)                │                          │
       └─────────────────────┘                                          │   asyncio.Lock           │
                                                                        │   (single-client per GPU)│
                                                                        └────────────┬─────────────┘
                                                                                     │
                                                                                     ▼
                            ┌──────────────────────────────────────────────────────────────┐
                            │                ServerState (server.py:39-92)                  │
                            │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  │
                            │  │ Opus Reader      │  │  Mimi (streaming)│  │  LMGen     │  │
                            │  │ Writer (sphn)    │  │  streaming_forever  │  streaming_ │  │
                            │  │ Opus⇄PCM 24kHz   │  │     (1)          │  │  forever(1)│  │
                            │  └────────┬─────────┘  └────────┬─────────┘  └─────┬──────┘  │
                            │           │                     │                  │         │
                            │           ▼                     ▼                  ▼         │
                            │     PCM frames        Mimi.encode(pcm)        LMGen.step    │
                            │     (1920 samples       → codes[B,K_user,1]    (CUDA Graphs) │
                            │      = 80ms)                                                  │
                            │           ┌──────────────────────────────────────────┐       │
                            │           │  Inference Loop (per 80ms frame)         │       │
                            │           │                                          │       │
                            │           │  1. opus_bytes  → PCM  (sphn.OpusReader) │       │
                            │           │  2. accumulate ≥1920 samples             │       │
                            │           │  3. mimi.encode → user codes [1,8,1]     │       │
                            │           │  4. lm_gen.step(user codes)              │       │
                            │           │       → tokens[1, 1+8, 1] (text+Moshi)   │       │
                            │           │  5. mimi.decode(tokens[:,1:])  → PCM     │       │
                            │           │  6. opus encode → ws.send_bytes(\x01...) │       │
                            │           │  7. text → ws.send_bytes(\x02...)        │       │
                            │           └──────────────────────────────────────────┘       │
                            └──────────────────────────────────────────────────────────────┘
```

---

### 2.2 Streaming 추상화 — 핵심 설계 패턴

```
       StreamingModule (modules/streaming.py:54-211)
       ============================================
       모든 stateful 모듈의 베이스. KV 캐시, conv 캐시 등을 State에 보관.

       ┌──────────────────────────────────────────────────────────┐
       │  class State (abc):                                       │
       │    batch_size: int                                        │
       │    device: torch.device                                   │
       │    exec_mask: BoolTensor[B]   ◀── batched out-of-sync exec│   streaming.py:35
       │    reset(reset_mask)          per-batch-item reset        │   streaming.py:41
       │    set_exec_mask(exec_mask)                               │   streaming.py:38
       └──────────────────────────────────────────────────────────┘
                              │ ⊆
       ┌──────────────────────┴─────────────────────────────────────┐
       │  class StreamingModule(nn.Module, Generic[StateT]):         │
       │    _streaming_state: StateT | None                          │
       │    _streaming_detached: bool                                │
       │    streaming(B):  enter streaming mode                      │   streaming.py:131
       │    streaming_forever(B):  same, but no exit                 │   streaming.py:128
       │    reset_streaming(reset_mask)                              │   streaming.py:139
       │    set_exec_mask(mask)  ⟵ batch desync                     │   streaming.py:183
       └──────────────────────────────────────────────────────────────┘
```

**Detached streaming의 묘미 (lm.py:579, 218):**
- **outer loop**: Temporal Transformer가 시간축으로 streaming (1 step = 80 ms 1 프레임).
- **inner loop**: Depth Transformer가 매 시간 step마다 코드북축으로 K번 streaming (1 step = 1 codebook).
- 둘은 서로 다른 reset 주기 → `set_streaming_detached(True)`로 부모의 streaming context를 무시.

---

### 2.3 추론 최적화

```
                  RingKVCache (transformer.py:196-...)
                  ====================================
                  KV cache를 길이 capacity의 ring buffer로 관리.
                  Cuda Graph와 호환되도록 동일 텐서 shape 유지.

       cache[2, B, H, capacity, D]    ┐ slot 0 = key, slot 1 = value
              end_offset[B] ──────────┘  insert position % capacity
       write: scatter at positions = (end_offset + [0..T)) % capacity   transformer.py:241-249


                  CUDAGraphed (utils/compile.py)
                  ==============================
       lm.py:630-632, compression.py:225-229
       - lm_model.forward_text 와 depformer_step 를 CUDA Graph로 캡쳐
       - mimi.encoder/decoder, encoder_transformer/decoder_transformer 도 캡쳐
       - 매 step CPU↔GPU 디스패치 오버헤드 제거 → 200ms p99 latency 달성 (README.md:40)
```

**워밍업 (server.py:62-72):** 모델을 처음 4 chunk 돌려서 CUDA Graph 캡쳐 + cuDNN auto-tune 완료시키고 첫 사용자 응답 지연 제거.

---

## 3. 생소한 개념 정리

| 개념 | 설명 | 코드 레퍼런스 |
|------|------|----|
| **Full-Duplex** | 전화처럼 양쪽이 **동시에** 말하고 들을 수 있는 통신. Turn-based STT→LLM→TTS와 달리 끼어들기/맞장구/침묵을 자연스럽게 처리. Moshi는 user 스트림과 자기 스트림을 **함께 모델링**하므로 별도 VAD/턴 매니저가 필요 없음. | lm.py:50 |
| **RQ-Transformer** | "Residual Quantizer Transformer". K개 잔차 코드북을 갖는 토큰을 모델링하기 위해 **시간축 큰 Transformer + 코드북축 작은 Transformer**의 2단 구조. MusicGen/AudioCraft 계보 + Depth-wise 확장. | lm.py (전체) |
| **Inner Monologue** | Moshi가 발화 audio token과 함께 자신이 말할 텍스트도 예측. 텍스트가 "계획"으로 작동해 long-range coherence/factuality 향상. | README.md:36-37, lm.py:139, 745 |
| **Delayed Streams Modeling** | 코드북마다 시간 오프셋을 다르게 두는 학습 트릭. semantic 코드북을 먼저 예측하고 그 결과를 acoustic 코드북 예측에 사용. AudioCraft에서 음악용으로 시작, Moshi/Hibiki에서 음성/번역에 응용. | lm_utils.py:9-38, loaders.py:118 |
| **Mimi** | Kyutai의 streaming neural audio codec. 24 kHz → 12.5 Hz / 1.1 kbps. SoundStream/EnCodec 계열 + Transformer + WavLM distillation + 적대적 학습만 사용. | compression.py:105 |
| **SEANet** | 신경 오디오 압축용 fully-convolutional encoder/decoder (Tagliasacchi et al). residual block + 다단 스트라이드. Mimi의 backbone. Moshi에서는 **causal & streaming-safe**한 변형 사용. | modules/seanet.py:96 |
| **Split RVQ** | semantic 정보용 첫 1개 코드북 + acoustic 잔차 7개 코드북을 **다른 projection으로 분리**해 학습 안정화. semantic은 WavLM distillation으로 강제. | quantization/vq.py:170-322 |
| **WavLM Distillation** | Microsoft의 self-supervised speech encoder. Mimi의 첫 코드북은 WavLM 임베딩과 cosine 유사도가 높도록 distill → 한 codec에 semantic+acoustic 통합. | README.md:58 |
| **Causal / Streaming Conv** | 좌측 패딩만 사용하는 1D conv. 추론 시 마지막 (kernel-1) 샘플을 버퍼링해서 다음 청크에 이어붙임 → 음향 정렬을 깨지 않고 무한 스트림 처리. | modules/conv.py |
| **CFG (Classifier-Free Guidance)** | 조건 c와 null c̄로 두 번 forward한 뒤 `logits = logits_null + γ·(logits − logits_null)`로 조건 영향을 증폭. | lm.py:600-603, 714-733 |
| **LoRA** | Low-rank adapters로 fine-tune. Moshi는 LoRA 가중치를 별도 safetensors로 받아 inference 시 fuse 가능. | modules/lora.py, loaders.py:486-514 |
| **RoPE** | Rotary Position Embedding (Su et al). attention head 차원을 짝지어 complex 회전으로 위치 인코딩. | modules/rope.py |
| **RMSNorm-f32** | LayerNorm 대신 평균 제거 없이 RMS만 정규화. FP32에서 계산 후 입력 dtype으로 복원 → bf16 추론 안정성. | transformer.py:46-77 |
| **SwiGLU Gating** | `silu(W1 x) * (W2 x)` FFN. LLaMA 계열 표준. | modules/gating.py |
| **Opus** | 저지연 음성 코덱(IETF RFC 6716). 자세히: [`01-opus-codec.md`](01-opus-codec.md) | server.py:78, 122 |
| **CUDA Graph** | 반복 호출되는 GPU 워크로드의 디스패치를 한번 캡쳐해 재실행. | lm.py:630-632 |
| **safetensors** | HuggingFace의 안전/빠른 weight 포맷 (memory-mapped, no pickle). | loaders.py:319, 357 |
| **Depthformer "weights_per_step"** | Depth Transformer가 K codebook 각각에 대해 **다른 weight set**을 사용. 첫 코드북 vs 7번째 코드북이 본질적으로 다른 분포라 효과적. | loaders.py:117, lm.py:172 |
| **`exec_mask`** | 한 배치 내 시퀀스마다 동기화 안 된 채 step. 어떤 슬롯은 "지금은 정지" 시키고 다른 슬롯만 진행 → multi-tenant 서빙. | streaming.py:38, lm.py:544-547 |

---

## 4. Latency Budget (200 ms 달성의 비밀)

```
   사용자 음성 시작 ─────────────────────────────────────────▶ Moshi 음성 시작
                                                                ▲
   ┌─────────┬───────────┬───────────────────┬─────────────────┐
   │ 80 ms   │ 80 ms     │  ~30 ms           │   80 ms acoustic
   │ 1 frame │ Mimi enc  │  Moshi LM step    │   delay (delay=1 on
   │ buffer  │ + LM      │  (Temporal +      │   acoustic codebooks)
   │ (mic)   │ context   │   Depth × 8)      │
   └─────────┴───────────┴───────────────────┴─────────────────┘
   = 이론치: 80 (frame) + 80 (acoustic delay) = 160 ms        README.md:40
   = 실측치: ~200 ms on L4 GPU
```

기여 요인:
1. **Mimi의 12.5 Hz 프레임** — 자기회귀 step 수가 codec frame rate에 비례 → 50 Hz보다 4× 적음.
2. **Depth Transformer 분리** — 큰 LM은 시간당 1번만, K=8개 codebook은 작은 Transformer에서 처리.
3. **CUDA Graph + RingKVCache** — Python/dispatch 오버헤드 zero, 메모리 할당 zero.
4. **Streaming Causal Conv** — Mimi encoder/decoder가 chunk 단위로 점진 처리.
5. **Opus + WebSocket** — 네트워크 페이로드 최소화.

---

## 5. 진입점 (Entry Points)

- `moshi/models/lm.py:49` — `LMModel` 메인 클래스
- `moshi/models/lm.py:556` — `LMGen` 추론 wrapper
- `moshi/models/compression.py:105` — `MimiModel`
- `moshi/modules/transformer.py` — `StreamingTransformer`, `RingKVCache`
- `moshi/modules/streaming.py:54` — `StreamingModule` 베이스
- `moshi/server.py:94` — `recv_loop` (WebSocket 핸들러)
- `moshi/models/loaders.py:90` — `_lm_kwargs` (모든 LM 하이퍼파라미터)

## 6. 핵심 요약

- **Moshi는 STT → LLM → TTS 파이프라인이 아니라 단일 end-to-end 모델**. user/Moshi 오디오를 두 개의 병렬 token 스트림으로, 자기 발화 텍스트를 inner monologue 스트림으로 동시에 autoregressive하게 모델링.
- **RQ-Transformer = Temporal(7B, 12.5 Hz) + Depth(small, 코드북축)** — Mimi의 K=8 RVQ 토큰을 깊이축으로 처리하면서 시간축은 한 번만 큰 모델을 통과.
- **시스템 측은 "StreamingModule" 추상화**가 핵심. State + exec_mask + detached streaming으로 inner/outer loop, batched desync, CUDA Graph 호환을 모두 같은 인터페이스에서 처리.
- **3개 백엔드(PyTorch/MLX/Rust)** 모두 동일 WebSocket 프로토콜로 동일 웹 UI와 통신 → 클라이언트-서버 분리가 깔끔.

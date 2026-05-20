# Neural Codec · RVQ · Moshi 아키텍처 · Streaming 메커니즘

> 대화 정리 노트
> 주제: Neural codec과 RVQ 기본 개념부터 Moshi의 full-duplex 아키텍처, Mimi encoder streaming 메커니즘까지
> 작성일: 2026-05-19

---

## 목차

1. [Neural Codec과 RVQ 기본 개념](#1-neural-codec과-rvq-기본-개념)
2. [Streaming Neural Codec의 Time Sync](#2-streaming-neural-codec의-time-sync)
3. [RVQ Flow Chart](#3-rvq-flow-chart)
4. [음성 대화 시스템 분류 (Half/Pseudo/True Full-duplex)](#4-음성-대화-시스템-분류)
5. [Moshi 전체 아키텍처](#5-moshi-전체-아키텍처)
6. [Inner Monologue 메커니즘](#6-inner-monologue-메커니즘)
7. [Codebook이란? (VQ 기본)](#7-codebook이란-vq-기본)
8. [Mimi Encoder → RQ-Transformer 전체 흐름](#8-mimi-encoder--rq-transformer-전체-흐름)
9. [Streaming Chunker와 12.5 Hz 해석](#9-streaming-chunker와-125-hz-해석)
10. [Conv State Cache의 깊이](#10-conv-state-cache의-깊이)
11. [전체 레퍼런스 모음](#11-전체-레퍼런스-모음)

---

## 1. Neural Codec과 RVQ 기본 개념

### Neural Codec

기존 신호처리 기반 codec (MP3, Opus 등)을 신경망으로 대체한 audio codec.
일반적으로 **encoder-quantizer-decoder** 구조로, 시간 도메인 waveform을 압축된 latent로 인코딩하고
이를 discrete token으로 양자화한 뒤 복원.

대표 모델:
- **SoundStream** (Google, 2021)
- **EnCodec** (Meta, 2022)
- **DAC** (Descript, 2023)

학습은 보통 reconstruction loss + adversarial loss (HiFi-GAN style discriminator) + perceptual loss 조합.

핵심 의의는 단순 압축을 넘어서, **discrete audio token**을 만들어내 LLM 기반 audio 생성
(AudioLM, VALL-E, MusicGen 등)의 vocabulary 역할을 한다는 점.

### RVQ (Residual Vector Quantization)

Vector Quantization을 **여러 단계에 걸쳐 잔차(residual)에 반복 적용**하는 기법.

**동작 방식:**
1. Encoder output `z`를 첫 번째 codebook `C₁`으로 quantize → `ẑ₁`
2. Residual `r₁ = z − ẑ₁`을 두 번째 codebook `C₂`로 quantize → `ẑ₂`
3. `r₂ = r₁ − ẑ₂`를 `C₃`으로 quantize, ... N번 반복
4. 최종 표현: `ẑ = Σᵢ ẑᵢ`

**왜 쓰는가:**
- Single VQ로 같은 표현력을 얻으려면 codebook size가 exponential하게 커져야 함 (`K^N` vs `N·K`).
  RVQ는 이를 선형으로 분해.
- **Bitrate scalability**: inference 시 앞쪽 k < N개 stage만 써도 coarse reconstruction이 가능해서,
  한 모델로 여러 bitrate 지원.
- Codebook collapse 완화에 유리 (각 stage가 다른 scale의 정보를 학습).

### 둘의 결합

Neural codec에서 RVQ는 사실상 표준. EnCodec의 경우 8개 stage × 1024 codebook → 75Hz frame rate에서
6 kbps 같은 식. 이렇게 나온 token sequence는 **flatten / parallel** 방식으로 LLM이 모델링하게 되는데,
MusicGen의 delayed pattern이나 VALL-E의 NAR+AR 구조가 이 RVQ 계층 구조를 어떻게 풀어내느냐의 design choice.

---

## 2. Streaming Neural Codec의 Time Sync

Streaming neural codec의 time sync는 크게 **frame rate 정합**, **causal architecture**, **buffer/latency 관리** 세 축.

### Frame rate가 sync의 기준점

Neural codec은 일정한 downsampling ratio로 동작.
- EnCodec 24kHz 모델: stride 320 → **75 Hz frame rate** (한 frame = 13.3 ms)
- 24000 samples/sec ÷ 320 = 75 tokens/sec
- N개 RVQ stage면 75 × N tokens/sec
- Mimi (Moshi): 12.5 Hz, DAC 24kHz: 75 Hz, SoundStream: 보통 50-75 Hz

LLM이 audio token을 생성할 때도 이 frame rate가 wall-clock과 정렬되어야 real-time이 성립.
즉 **token generation throughput ≥ frame rate**가 필수 조건.

### Causal / streaming architecture

기본 EnCodec, DAC는 non-causal convolution (양방향 context)을 써서 그대로는 streaming 불가. Streaming variant:
- **Causal convolution**: padding을 left-only로 바꿈
- **Causal normalization**: LayerNorm/GroupNorm을 streaming-safe하게
- **No future lookahead**: Transformer block이 있다면 causal mask 적용

### Chunk 처리와 conv state caching

Streaming에서 conv를 chunk-by-chunk로 돌리려면 **kernel receptive field만큼의 과거 sample을 cache**해야
boundary가 끊기지 않음. 일종의 KV cache의 conv 버전.

```
chunk_t 입력 → [cached_left_context | chunk_t] → conv → output_t
                                                    ↓
                                            cache 업데이트 (오른쪽 끝 보관)
```

### Encoder ↔ Decoder ↔ LLM 사이의 sync

Full-duplex agent (Moshi 같은) 구조:
- Audio I/O는 hardware clock (sound card) 기준 → **이게 진짜 master clock**
- Codec frame rate는 이 clock에서 파생
- LLM은 **token rate를 따라잡아야** 함. 못 따라가면 underrun → 출력 끊김

### RVQ stage와 sync

RVQ N개 stage가 한 frame에 묶여 있으므로, **한 frame이 완성되려면 N개 token이 다 나와야** 함.
- **Flatten** (VALL-E 식): N stage × T frames를 일렬로 → time resolution 떨어짐
- **Parallel / delayed pattern** (MusicGen): stage들을 같은 step에 병렬 생성하되 살짝 어긋나게
- **Semantic + acoustic 분리** (Moshi/Mimi): semantic token 1개 + acoustic RVQ token 7개를 한 step에 같이

Mimi가 12.5 Hz로 내려간 게 결정적인데, GPU에서 LLM이 12.5 step/sec (80ms budget per step)면 충분히 따라잡을 수 있기 때문.

---

## 3. RVQ Flow Chart

```
            ┌─────────────────────┐
            │  Encoder output z   │
            │  (Continuous latent)│
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐         ┌─────────────────────┐
            │     Codebook C₁     │────────▶│   r₁ = z − ẑ₁       │
            │  ẑ₁ = nearest(z,C₁) │         │     Residual        │
            └──────────┬──────────┘         └─────────┬───────────┘
                       │                              │
                       │       ┌──────────────────────┘
                       │       ▼
                       │  ┌─────────────────────┐    ┌──────────────────┐
                       │  │     Codebook C₂     │───▶│ r₂ = r₁ − ẑ₂     │
                       │  │ ẑ₂ = nearest(r₁,C₂) │    │  Finer residual  │
                       │  └─────────────────────┘    └────────┬─────────┘
                       │                                       │
                       │  ┌─────────────────────────────────┐  │
                       │  │   ... repeat for C₃ ... C_N     │  │
                       │  └─────────────────┬───────────────┘  │
                       │                    │                  │
                       │  ┌─────────────────▼─────────────┐    │
                       │  │       Codebook C_N            │◀───┘
                       │  │  ẑ_N = nearest(r_{N−1}, C_N)  │
                       │  └────────────┬──────────────────┘
                       │               │
                       ▼               ▼
            ┌─────────────────────────────────────┐
            │     ẑ = ẑ₁ + ẑ₂ + … + ẑ_N            │ ───▶ Tokens [i₁, ..., i_N]
            │   Final quantized representation    │
            └─────────────────┬───────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Decoder → waveform  │
                  └──────────────────────┘
```

**핵심 포인트:**
- **세로 흐름 = 시간(stage) 진행**: 위에서 아래로 갈수록 더 미세한 정보. C₁은 coarse structure, C_N은 fine detail.
- **가로 흐름 = quantize → residual 계산**: 각 stage에서 codebook으로 양자화한 뒤, 그 오차를 다음 stage로 넘김.
- **결과는 합산**: 최종 표현 `ẑ = Σ ẑᵢ`. 각 codebook이 독립적으로 "이 latent의 어떤 측면을 책임진다"라고 분업.
- **Token sequence**: 각 stage의 codebook index `iₙ`이 곧 discrete token.
- **Bitrate scalability**: 추론할 때 앞쪽 k < N stage만 사용해서 `ẑ^(k) = Σᵢ₌₁ᵏ ẑᵢ`로 decode하면 low-bitrate 버전.

---

## 4. 음성 대화 시스템 분류

음성 대화 시스템 분류는 크게 **세 축**으로 나뉨.

### 축 1 — Duplex mode (시간적 행동)

| 모드 | 정의 | 대표 모델 |
|---|---|---|
| **Half-duplex (half-D)** | 사용자가 다 말한 뒤 agent가 응답. VAD로 turn 경계를 잘라줘야 함 | Qwen2.5-Omni, GLM-4-Voice, AudioGPT |
| **Pseudo full-duplex (PFD)** | 시분할 다중화 방식으로 동시성을 "흉내". 실제로는 빠른 chunk 단위 alternation | Freeze-Omni, VITA-1.5, DuplexCascade |
| **True full-duplex (TFD)** | 동시 listening/speaking, 실시간 출력 적응, interruption·backchannel·adaptive turn-taking 지원 | Moshi, dGSLM, Hertz-dev |

True full-duplex는 양방향 정보 흐름을 갖는 parallel encoding/decoding이 필수이며 실시간 제약(< 200ms) 만족 필요.

### 축 2 — Architecture (구현 방식)

- **Cascaded**: ASR → LLM → TTS pipeline
- **End-to-end (E2E)**: speech token 또는 latent를 직접 처리

### 축 3 — Synchronization (FD-SLM 분류, 가장 최근)

- **Engineered synchronization**: FSM이나 duplex controller로 turn-taking과 semantic arbitration 관리
  - 예시: FlexDuo의 ternary FSM, VITA-1.5의 dual LLM with shared KV cache, ASR 기반 Semantic VAD
- **Learned synchronization**: parallel stream을 모델이 직접 학습
  - 예시: Moshi가 대표

### Timeline 패턴 (텍스트 다이어그램)

```
Half-duplex:
  User:  ████████____________
  Agent: __________████████__
         <-- VAD gap -->

Pseudo full-duplex:
  User:  ██_███_██__████_____
  Agent: __█___█___█_____██__
         <-- chunked micro-turns -->

True full-duplex:
  User:  ████████████████████
  Agent: ____█___█_____█████_
         <-- overlapping, backchannels -->
```

---

## 5. Moshi 전체 아키텍처

### Moshi의 분류

| 축 | 분류 |
|---|---|
| **Duplex mode** | True full-duplex |
| **Architecture** | End-to-end |
| **Synchronization** | Learned synchronization |
| **Token representation** | Discrete (Mimi RVQ codec tokens) |

### 구조 개요

```
┌──────────────────┐                              ┌──────────────────┐
│  Mimi encoder    │                              │  Mimi decoder    │
│  User audio      │                              │  Moshi audio     │
│  24 kHz, causal  │                              │  24 kHz, causal  │
└────────┬─────────┘                              └────────▲─────────┘
         │                                                  │
         ▼                                                  │
┌──────────────────┐    ┌────────────────────────────┐    ┌──────────────────┐
│ User audio tokens│───▶│      RQ-Transformer        │───▶│ Moshi audio tokens│
│ 8 codebooks/step │    │  ┌──────────────────────┐  │    │ 8 codebooks/step │
│ 1 sem + 7 acoust │    │  │ Temporal Transformer │  │    │ Acoustic delay τ=1│
│ 12.5 Hz · 1.1 kbps│    │  │ 32 layers · 7B      │  │    └──────────────────┘
└──────────────────┘    │  │ Helium init          │  │              ▲
                        │  │ Time deps            │  │              │
                        │  └──────────┬───────────┘  │              │
                        │             │ context c_t  │              │
                        │             ▼               │              │
                        │  ┌──────────────────────┐  │              │
                        │  │  Depth Transformer   │  │──────────────┘
                        │  │  6 layers, 1024 dim  │  │
                        │  │  K=17 tokens / step  │  │
                        │  │  Inter-codebook deps │  │
                        │  └──────────┬───────────┘  │
                        └─────────────┼──────────────┘
                                      │
                                      ▼
                        ┌────────────────────────────┐
                        │  Inner monologue (text)    │
                        │  Time-aligned, leads audio │
                        │  by τ=1 (80 ms)            │
                        └────────────────────────────┘
```

### 주요 컴포넌트 정리

**Mimi codec (양옆)** — neural audio codec
- 24 kHz 오디오를 12.5 Hz representation, 1.1 kbps로 fully streaming
- Latency 80 ms (frame size)
- RVQ와 일반 VQ 결합 — RVQ로 acoustic token, VQ로 semantic token 학습
- 1 semantic VQ + 7 acoustic RVQ = 8 codebooks/step

**User audio tokens / Moshi audio tokens** — 두 stream이 parallel하게 모델링
- 사용자와 Moshi 각각의 audio를 분리 stream으로
- text token은 Moshi 자신의 발화에만 (inner monologue)

**RQ-Transformer** — Moshi의 핵심 아키텍처
- 시간 방향(Temporal Transformer, 32 layers, Helium에서 초기화)과 깊이 방향(Depth Transformer, 6 layers)으로 autoregression 분해
- Acoustic delay τ=1 (80 ms)로 theoretical 160 ms / practical 200 ms latency 달성
- K=17 (1 text + 8 user audio + 8 Moshi audio)

**Helium 7B** — 별도로 학습된 text LLM이 Temporal Transformer의 초기 weight
- 2.1T tokens 영문 텍스트로 학습된 7B Transformer
- RMSNorm, RoPE, SiLU activation의 GLU, 4096 context length

### Spoken QA 성능 (Inner Monologue 효과)

| 모델 설정 | WebQ | LlamaQ |
|---|---:|---:|
| Inner Monologue 없이 | 9% | 21% |
| Inner Monologue 적용 | 26.6% | 62.3% |

Spectron, SpeechGPT를 streaming 환경에서 능가.

---

## 6. Inner Monologue 메커니즘

### 핵심 아이디어

Moshi는 speech-to-speech 모델이지만 audio token만으로 일관된 speech 생성이 어려워서,
coarse-to-fine generation을 한 단계 더 확장. **text → semantic → acoustic** 계층 구조.

기존 hierarchy (AudioLM): `semantic → acoustic`
Moshi의 확장: `text → semantic → acoustic` (text가 가장 위)

### 12.5 Hz 정렬 메커니즘

각 80 ms frame이 다음을 포함:
- W_t row: 1개 text token (PAD, EPAD, 또는 word BPE piece)
- 1개 semantic token
- 7개 acoustic token

### Text 정렬 = Whisper word timestamps

- Whisper로 word-level timestamp 추출
- 각 word i의 start time tᵢ를 12.5 Hz framerate로 나눠 frame index 결정
- 그 frame에 word의 BPE token들을 배치

### 특수 토큰: PAD / EPAD

```
시간:     t=1  t=2   t=3    t=4  t=5    t=6   t=7   t=8
W_t row:  PAD  EPAD  "Hi"   PAD  EPAD   "fri" "end" PAD
```

- **PAD**: 침묵 채움 (말이 없는 frame)
- **EPAD**: end-of-pad, 단어 시작 직전에 삽입 ("이제 단어 시작한다" 신호)
- **Word BPE**: Whisper start timestamp ÷ 80 ms 위치에 배치

EPAD는 "PAD 끝났음 + 단어 시작" 결정을 두 단계로 분리해서 모델 학습 안정성 향상.

### Per-frame Autoregressive 생성 순서

각 frame s 안에서 Depth Transformer가 다음 순서로 17개 token을 autoregressive 생성:

```
W_s → S_s → A_s¹ → A_s² → A_s³ → A_s⁴ → A_s⁵ → A_s⁶ → A_s⁷
(text)(semantic)              (acoustic × 7)
```

Text가 제일 먼저 → semantic을 condition → semantic이 acoustic detail을 condition.

### Loss Weights

```
L_s = CE(l_{s,1}, W_s) + (1/Σα_k) Σ α_k CE(l_{s,k}, V_{s,k})
```

- α₁ = 1 (text)
- α₂ = 100 (semantic) ← **가장 높음**
- α₃..₈ = 1 (acoustic)

### 사용자 음성은 transcribe하지 않음

W_t는 **Moshi 자신의 발화만** 모니터링.
사용자 음성에 대해 별도 inner monologue 안 만듦 → 외부 ASR 의존하면 E2E 정신에 어긋남.

### Streaming TTS/ASR 파생

같은 모델에서 text-audio delay를 어떻게 주느냐에 따라:
- **Text가 audio를 lead** → streaming **TTS** (text 먼저 정하고 음성 만들기)
- **Audio가 text를 lead** → streaming **ASR** (음성 듣고 text 결정)

### Per-frame decomposition (vs. SpeechGPT/Spectron)

- Spectron/SpeechGPT: "전체 텍스트 다 만들고 → 그 다음 audio 생성" (block 모드, real-time 불가)
- Moshi: "한 frame씩 (text + audio 같이) 생성" → streaming 가능

### 실전 응용 시 주의점

Moshi에 외부에서 텍스트를 주입할 때:
- 80ms마다 ~20자(약 4-5 tokens)씩 드립피드
- 한꺼번에 300자를 부으면 inner monologue가 같은 token에 attention 고정되면서 degenerate
- 12.5 Hz라는 frame rate가 그냥 숫자가 아니라 실제 throughput constraint

---

## 7. Codebook이란? (VQ 기본)

### 정의

**Codebook = 학습 가능한 prototype vector들의 lookup table.**

- 학습 가능한 embedding matrix, shape `[K, D]`
- K = codebook size (e.g. 1024, 2048)
- D = embedding dim (e.g. 128, 256, 512)

### 동작

1. **Encoder가 continuous vector z ∈ ℝᴰ를 출력**
2. **Codebook의 K개 prototype 중 가장 가까운 것 (`argmin_j ‖z − c_j‖²`) 선택**
3. **그 index i가 discrete token, 그 vector c_i가 quantized output ẑ**

### 2D 직관: Voronoi Partition

```
       ▲ y
       │
   c₀  │  c₁     c₂   ← codebook prototypes
   •───┼───•─────•
       │      ×       ← input z (snaps to nearest c₂)
   c₃  │  c₄     c₅
   •───┼───•─────•
       │
       └─────────────▶ x

codebook이 공간을 K개 Voronoi cell로 분할
입력 vector는 자기 cell의 anchor (c_k)로 snap
```

### 왜 "code book"이라 부르나

신호처리에서 60년대부터 쓰던 용어:
- Lloyd algorithm, k-means, Linde-Buzo-Gray algorithm이 모두 같은 개념
- 각 entry가 "code word", 전체 table이 "code book"
- VQ-VAE에서는 PyTorch `nn.Embedding` 모듈로 codebook 구현

### 기본 VQ 코드 (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int, commitment_weight: float = 0.25):
        super().__init__()
        # Codebook: K rows of D-dim vectors. This IS the codebook.
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.commitment_weight = commitment_weight

    def forward(self, z: torch.Tensor):
        # Compute squared L2 distance from each z to each code
        d = (
            z.pow(2).sum(dim=-1, keepdim=True)         # [B, 1]
            - 2 * z @ self.codebook.weight.t()          # [B, K]
            + self.codebook.weight.pow(2).sum(dim=-1)   # [K]
        )

        # Nearest-neighbor lookup → discrete token index
        indices = d.argmin(dim=-1)                      # [B]

        # Retrieve quantized vector from the codebook
        z_q = self.codebook(indices)                    # [B, D]

        # Losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices, loss
```

**핵심 줄들:**
- `self.codebook = nn.Embedding(num_codes, dim)` — 이게 바로 codebook
- `d = ...` — 모든 z와 모든 codebook entry 사이의 squared L2 distance를 batched 계산
- `indices = d.argmin(dim=-1)` — discrete token (EnCodec/Moshi에서 LLM이 다루는 audio token)
- `z_q = z + (z_q - z).detach()` — **straight-through estimator**.
  argmin은 미분 불가능 → forward는 quantize, backward는 identity로 처리.

### Residual VQ 확장

```python
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_stages: int, num_codes: int, dim: int):
        super().__init__()
        self.stages = nn.ModuleList([
            VectorQuantizer(num_codes, dim) for _ in range(num_stages)
        ])

    def forward(self, z: torch.Tensor):
        residual = z
        z_q_total = torch.zeros_like(z)
        all_indices = []
        total_loss = 0.0

        for vq in self.stages:
            z_q_n, idx_n, loss_n = vq(residual)
            z_q_total = z_q_total + z_q_n
            residual = residual - z_q_n
            all_indices.append(idx_n)
            total_loss = total_loss + loss_n

        tokens = torch.stack(all_indices, dim=0)
        return z_q_total, tokens, total_loss
```

### EnCodec 실제 사용 예

```python
from encodec import EncodecModel
import torchaudio

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps → 8 codebooks/frame

wav, sr = torchaudio.load("speech.wav")
wav = wav.unsqueeze(0)

encoded = model.encode(wav)
codes = encoded[0][0]  # [B, n_codebooks, T_frames]
# 예: torch.Size([1, 8, 187])  — 8 RVQ stages, 187 frames
```

### Codebook의 디테일들

1. **Codebook collapse**: 일부 codebook vector가 한 번도 사용되지 않으면 "dead". DAC가 거의 100% 사용률 달성.
2. **EMA update**: gradient 대신 exponential moving average로 codebook 업데이트가 더 안정적. SoundStream/EnCodec/DAC 모두 EMA 사용.
3. **Three loss terms**: reconstruction + codebook + commitment (β 가중치).
4. **K vs D 선택**: K 너무 크면 sparse, 너무 작으면 capacity 부족. Audio codec은 보통 K=1024 또는 2048, D=128~256.

---

## 8. Mimi Encoder → RQ-Transformer 전체 흐름

```
┌─────────────────────────────────────────┐
│  Raw waveform · user microphone         │
│  24 kHz · mono · float32 · [B, 1, T]    │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  Streaming chunker                       │
│  80 ms chunks = 1920 samples per frame   │
└──────────────────┬──────────────────────┘
                   ▼
╔═══════════════════════════════════════════════════════════════╗
║  Mimi encoder · causal · streaming         SEANet + Transformer║
║                                                                ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  Initial 1D causal conv                                  │  ║
║  │  [B, 1, T] → [B, C₀, T]                                  │  ║
║  └────────────────────────┬────────────────────────────────┘  ║
║                           ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  4 × SEANet residual blocks · strided downsampling       │  ║
║  │  stride factors (4, 5, 6, 8) → total downsample ×960     │  ║
║  │  24000 Hz → 25 Hz · all convolutions causal              │  ║
║  │  block = dilated conv + ELU + skip connection            │  ║
║  └────────────────────────┬────────────────────────────────┘  ║
║                           ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  Final 1D conv · stride 2                                │  ║
║  │  25 Hz → 12.5 Hz · D = 512                               │  ║
║  └────────────────────────┬────────────────────────────────┘  ║
║                           ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  8-layer Transformer · causal attention                  │  ║
║  │  refines per-frame latent · [B, 12.5 Hz frames, 512]     │  ║
║  └────────────────────────┬────────────────────────────────┘  ║
╚════════════════════════════╪═══════════════════════════════════╝
                             ▼
┌─────────────────────────────────────────┐
│  Continuous latent z                     │
│  [B, 512] per frame · 12.5 Hz            │
└──────────────────┬──────────────────────┘
                   ▼
╔═══════════════════════════════════════════════════════════════╗
║  Mimi RVQ · split residual VQ (semantic + acoustic)            ║
║                                                                ║
║  ┌─────────────────────────┐ ┌─────────────────────────────┐  ║
║  │  Semantic VQ · stage 1  │ │  Acoustic RVQ · stages 2…8  │  ║
║  │  K=2048                  │ │  7 stages · K=2048 each      │  ║
║  │  distilled from WavLM    │ │  → tokens a_t¹ … a_t⁷        │  ║
║  │  → token s_t (linguistic)│ │  applied to residual r₁      │  ║
║  │  residual r₁ = z − ẑ_sem │ │                              │  ║
║  └────────────┬─────────────┘ └─────────────┬───────────────┘  ║
║               ▼                              ▼                  ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │  Per-frame discrete tokens · 8 codebooks × 12.5 Hz · 1.1kbps│  ║
║  │  [ s_t · a_t¹ · a_t² · a_t³ · a_t⁴ · a_t⁵ · a_t⁶ · a_t⁷ ]│  ║
║  └────────────────────────┬────────────────────────────────┘  ║
╚════════════════════════════╪═══════════════════════════════════╝
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Token embedding · acoustic delay · stream sum               │
│  each codebook k has its own embedding table E_k → ℝ^4096    │
│  apply acoustic delay τ = 1 frame to acoustic tokens         │
│  sum 17 embeddings: 1 text + 8 user audio + 8 Moshi audio    │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Temporal Transformer · 32 layers · 7B · Helium init         │
│  input: summed embedding of frame t · shape [B, 4096]        │
│  output: context vector c_t · [B, 4096]                      │
│  one step per 80 ms frame · KV cache for streaming           │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  → Depth Transformer (next step)                             │
│  conditioned on c_t, generates W_s → S_s → A_s¹ … A_s⁷       │
│  9 autoregressive sub-steps within one 80 ms frame           │
└─────────────────────────────────────────────────────────────┘
```

### Mimi의 핵심: WavLM Distillation

첫 번째 RVQ stage가 일반 RVQ가 아니라 "semantic"을 책임지는 특수 VQ:
- WavLM teacher의 hidden representation에 first-level codebook을 align시키는 distillation으로 학습
- WavLM: 16kHz waveform → 50Hz × 1024-dim
- Mimi: 24kHz waveform → 12.5Hz × 512-dim
- 학습 중에 WavLM target을 12.5 Hz로 average pooling (stride 4, kernel 8)으로 맞춤
- 그래서 token s_t는 "이 frame에서 어떤 linguistic 내용이 발화되었나"를 압축

### 전체 데이터 shape 흐름 요약

| Stage | Shape | Rate |
|---|---|---|
| Raw audio | `[B, 1, T]` | 24000 Hz |
| After SEANet | `[B, C, T/960]` | 25 Hz |
| After stride-2 conv | `[B, 512, T/1920]` | 12.5 Hz |
| After Mimi Transformer | `[B, n_frames, 512]` | 12.5 Hz |
| RVQ tokens | `[B, 8, n_frames]` (int) | 12.5 Hz |
| Embedded + summed | `[B, n_frames, 4096]` | 12.5 Hz |
| Temporal Transformer out | `[B, n_frames, 4096]` | 12.5 Hz |

---

## 9. Streaming Chunker와 12.5 Hz 해석

### Streaming chunker는 STFT의 window/hop과 다름

Mimi는 **stride 기반 non-overlapping chunker** — hop이 곧 chunk size이고, overlap이 0%.

```
chunk_size = 1920 samples (= 80 ms at 24 kHz)
hop        = 1920 samples (same as chunk_size)
overlap    = 0 samples
```

### A. STFT 모델 (비교용, Mimi는 이렇게 하지 않음)

전통적 STFT-기반 audio 처리:
- window=512, hop=128 → **75% overlap**
- 각 sample이 4개 frame에 중복 등장
- 같은 신호를 여러 번 재계산 (SED, source separation에서 일반적)

### B. Mimi의 stride chunker (실제 동작)

```python
# Pseudocode for Mimi streaming chunker
buffer = []
for new_samples in mic_stream():       # mic gives e.g. 480 samples per callback
    buffer.extend(new_samples)
    while len(buffer) >= 1920:
        chunk = buffer[:1920]          # take exactly one frame's worth
        buffer = buffer[1920:]         # advance hop = chunk size
        token_frame = mimi_encoder.step(chunk)   # produces 1 frame of 8 tokens
        yield token_frame
```

마이크 callback이 480-sample(20ms)이든 1024든 상관없이, 내부 buffer에 쌓이다가
**1920 sample이 모일 때마다 정확히 한 번 encoder step을 호출**.

### C. Chunk 경계의 conv 처리 → State Cache

각 causal 1D conv에 **"왼쪽 context cache"**를 유지:

```python
class StreamingCausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.cache = None

    def step(self, chunk):
        if self.cache is None:
            self.cache = torch.zeros(*chunk.shape[:2], self.kernel_size - 1)

        x = torch.cat([self.cache, chunk], dim=-1)
        out = self.conv(x)
        self.cache = x[..., -(self.kernel_size - 1):]
        return out
```

**결과적으로 streaming inference의 출력이 batch inference의 출력과 비트 단위로 동일** ("perfect streaming").

### D. "12.5 Hz" 해석

세 가지 동등한 해석:

| 표현 | 의미 |
|---|---|
| **12.5 Hz** | "초당 12.5개의 token frame이 나온다" — wall-clock 기준 출력 속도 |
| **80 ms / frame** | 한 frame이 차지하는 시간 |
| **stride 1920 @ 24 kHz** | 입력 1920 sample마다 출력 1 frame |

계산: `24000 Hz / 1920 stride = 12.5 Hz`

**중요:** 이건 어떤 연속 신호의 sampling rate가 **아님**. 그냥 encoder의 출력 token이
시간상 등간격으로 12.5 Hz 간격을 갖는다는 뜻. 사실상 "frame rate"라는 단어가 더 정확.

### Update 방식

새 chunk가 들어올 때마다:
1. **입력 시간 +80 ms** (1920 samples 누적)
2. **출력 token frame +1** (codebook 8개 token)
3. **모든 conv layer cache가 자기 stride만큼 sliding update**
4. **encoder Transformer는 KV cache에 한 step 추가**

매 80ms마다 정확히 이 4가지가 일제히 한 칸씩 전진.

### STFT vs Mimi 차이의 본질

- **STFT**: 각 frame이 독립적으로 spectral content를 추출하기 위해 window를 겹쳐서 frequency leakage를 줄이고 시간 해상도 보강
- **Mimi**: temporal conv가 receptive field 안의 모든 sample을 어차피 보기 때문에, chunk끼리 겹칠 필요가 없고 state cache 한 줄로 해결

### 실전 디테일

**Latency = 80 ms? 정확히는 아님.**
"80 ms"는 **algorithmic latency = frame size**. 실제로는 encoder의 receptive field가 80ms보다 큼 (dilated conv 때문).
다만 causal cache 덕분에 새 80 ms chunk를 받으면 즉시 새 token frame을 뽑을 수 있으므로 **추가 input-output delay는 0**.

**마이크 callback과 mismatch.**
OS 마이크 callback은 보통 10ms, 20ms 단위. Mimi의 80ms frame과 안 맞으면 buffer 패턴으로 흡수.

**Transposed conv decoder의 어려움.**
Decoder는 transposed conv를 써서 stride만큼 sample을 "내놓아야" 함. Mimi decoder는 input은 12.5 Hz token이지만
output은 80ms 단위 1920 sample을 한 번에 토해냄.

---

## 10. Conv State Cache의 깊이

### 결론: Cache의 "깊이"는 layer마다 다름

**한 줄 답:** Cache는 "전체적으로 N 프레임"이 아니라 **각 conv layer가 자기 (kernel_size − 1) × dilation 만큼의
left context를 sample 단위로 저장**. 모든 layer의 cache를 합한 총 receptive field가 encoder의 "effective lookback".

### A. SEANet encoder의 layer 구조

```
[Initial conv k=7]
  → [ResBlock(dilation=1) → Down stride=4]
  → [ResBlock(dilation=1) → Down stride=5]
  → [ResBlock(dilation=1) → Down stride=6]
  → [ResBlock(dilation=1) → Down stride=8]
  → [Final conv stride=2]
  → [Transformer 8 layers]
```

각 ResBlock 안에는 kernel_sizes=(3, 1), dilations=(1, 1)의 두 conv.

### B. Layer별 cache 크기 (sample 단위)

**각 causal conv의 cache 크기 = `(kernel_size − 1) × dilation`**

여기가 헷갈리기 쉬운 핵심: cache는 **frame 수가 아니라 sample 수로 셈**
(정확히는 그 layer의 입력 frame rate 기준 sample 수).

| Layer | Kernel | Dilation | Cache (그 layer 입력 단위) | Input rate |
|---|---:|---:|---:|---:|
| Initial conv | 7 | 1 | 6 | 24000 Hz |
| ResBlock1 conv1 | 3 | 1 | 2 | 24000 Hz |
| ResBlock1 conv2 | 1 | 1 | 0 | 24000 Hz |
| Down stride=4 | 8 | 1 | 7 | 24000 Hz |
| ResBlock2 conv1 | 3 | 1 | 2 | 6000 Hz |
| ResBlock2 conv2 | 1 | 1 | 0 | 6000 Hz |
| Down stride=5 | 10 | 1 | 9 | 6000 Hz |
| ResBlock3 conv1 | 3 | 1 | 2 | 1200 Hz |
| ResBlock3 conv2 | 1 | 1 | 0 | 1200 Hz |
| Down stride=6 | 12 | 1 | 11 | 1200 Hz |
| ResBlock4 conv1 | 3 | 1 | 2 | 200 Hz |
| ResBlock4 conv2 | 1 | 1 | 0 | 200 Hz |
| Down stride=8 | 16 | 1 | 15 | 200 Hz |
| Final conv stride=2 | (~4) | 1 | 3 | 25 Hz |

각 layer는 자기 cache를 **자기 입력 rate 기준 sample로** 들고 있음.
Deep layer일수록 cache 크기는 작지만 한 sample이 표현하는 시간이 길어짐.

### C. Receptive Field 환산 (wall-clock 시간)

그 layer 입력 rate가 r일 때, 그 cache의 1 sample은 24000/r 개의 입력 sample을 잡고 있음.

| Layer | Cache (자기 단위) | 환산 (24kHz sample) | 환산 (ms) |
|---|---:|---:|---:|
| Initial conv | 6 | 6 | 0.25 |
| ResBlock1 conv1 | 2 | 2 | 0.08 |
| Down ×4 | 7 | 7 | 0.29 |
| ResBlock2 conv1 | 2 | 8 | 0.33 |
| Down ×5 | 9 | 36 | 1.5 |
| ResBlock3 conv1 | 2 | 40 | 1.67 |
| Down ×6 | 11 | 220 | 9.17 |
| ResBlock4 conv1 | 2 | 240 | 10.0 |
| Down ×8 | 15 | 1800 | 75.0 |
| Final stride=2 | 3 | 2880 | 120.0 |
| **합계** | | **5239** | **≈218 ms** |

→ **순수 conv state cache 기준 약 220 ms ≈ 2.75 frames** (80 ms 기준).

### D. Frame 단위로 다시 본다면

질문의 "몇 프레임?"에 대한 정확한 답:
- **출력 frame 기준 약 3 frames**
- **wall-clock 220-300 ms**

중요한 점:

1. **모든 cache가 같은 frame 단위로 정렬되어 있지 않다.**
   Deep layer일수록 자기 frame이 길어서 "이전 N 프레임"이라는 표현 자체가 layer별로 의미가 달라짐.

2. **Frame이 아니라 sample을 저장한다.**
   Encoder가 12.5 Hz token을 만들지만, cache는 그 token이 아니라 **각 layer의 중간 activation tensor**를 저장.

3. **새 chunk가 들어오면 모든 cache가 동시에 sliding update.**
   매 step마다 **모든 14개 conv layer cache가 일제히 자기 stride만큼 shift**.

### E. 의미 있게 비교할 단위 — "effective lookback"

| 질문 | 답에 쓸 단위 |
|---|---|
| "Streaming inference 시작 후 몇 ms 후에 출력이 batch와 동일해지나?" | **conv state cache 총합 ≈ 220 ms** (warmup latency) |
| "한 번에 메모리에 몇 byte의 cache가 떠있나?" | **각 layer cache 크기 × channel × dtype의 합** |
| "이 한 token이 입력의 어느 시간 범위를 반영하나?" | **theoretical receptive field ≈ 1초 이상** |

**Warmup의 의미** — Mimi가 처음 시작될 때, 첫 chunk의 cache는 zero padding이라 처음 몇 frame은 정확하지 않을 수 있음.
대략 3 frame (≈240 ms) 정도 지나야 모든 cache가 실제 데이터로 채워지고 출력이 "정상화".

### F. Moshi 전체 streaming state 종류

위의 cache는 **conv layer의 state cache**. 이와 별개로:

1. Mimi encoder의 conv state caches (≈220 ms)
2. Mimi encoder의 Transformer KV cache (~250–750 frames)
3. Mimi decoder의 conv state caches (encoder와 대칭)
4. Mimi decoder의 Transformer KV cache
5. RQ-Transformer (Temporal + Depth)의 KV cache

다섯 종류의 cache가 동시에 streaming 중에 유지되고, 각자 update 주기와 크기가 다름.

### 한 줄 정리

| 질문 | 답 |
|---|---|
| Cache가 몇 프레임 누적? | **단일 숫자가 아님.** Conv state cache는 layer별 (k-1)×d sample 단위로 분산 저장 |
| Frame 단위로 환산하면? | 모든 conv cache 총합 ≈ **220 ms ≈ 약 3 frames** at 80 ms/frame |
| Update 방식? | 매 80 ms마다 모든 cache가 자기 stride만큼 sliding (FIFO) |
| 가장 큰 비중 차지? | 깊은 layer의 down-stride conv (cache 1 sample = wall-clock 75–120 ms씩 차지) |

---

## 11. 전체 레퍼런스 모음

### Moshi & Mimi
- Moshi paper (arXiv): https://arxiv.org/abs/2410.00037
- Moshi paper HTML v2: https://arxiv.org/html/2410.00037v2
- Kyutai PDF (full): https://kyutai.org/Moshi.pdf
- Moshi GitHub: https://github.com/kyutai-labs/moshi
- HuggingFace Moshi docs: https://huggingface.co/docs/transformers/model_doc/moshi
- HuggingFace Mimi model card: https://huggingface.co/kyutai/mimi
- HuggingFace Mimi docs: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mimi.md
- HuggingFace transformers Mimi 구현: https://github.com/huggingface/transformers/tree/main/src/transformers/models/mimi
- Moshi fine-tune: https://github.com/kyutai-labs/moshi-finetune
- Kyutai org page: https://kyutai.org/

### Neural Codec 원논문
- **SoundStream** (Zeghidour et al., 2021): https://arxiv.org/abs/2107.03312
- **EnCodec** (Défossez et al., 2022): https://arxiv.org/abs/2210.13438
- **DAC** (Kumar et al., NeurIPS 2023): https://arxiv.org/abs/2306.06546
- **SEANet** (Tagliasacchi et al., 2020): https://arxiv.org/abs/2009.02095
- EnCodec Wikipedia: https://en.wikipedia.org/wiki/EnCodec
- EnCodec GitHub (Meta): https://github.com/facebookresearch/encodec
- DAC GitHub (Descript): https://github.com/descriptinc/descript-audio-codec
- HuggingFace EnCodec docs: https://huggingface.co/docs/transformers/model_doc/encodec
- HuggingFace EnCodec source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py
- FunCodec SEANet encoder (잘 주석된 fork): https://github.com/modelscope/FunCodec/blob/master/funcodec/models/encoder/seanet_encoder.py
- Audiocraft EnCodec docs: https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md

### VQ / RVQ
- **VQ-VAE** (van den Oord et al., NeurIPS 2017): https://arxiv.org/abs/1711.00937
- **VQ-VAE-2** (Razavi et al., 2019): https://arxiv.org/abs/1906.00446
- **Linde-Buzo-Gray** (1980): doi.org/10.1109/TCOM.1980.1094577
- HuggingFace blog "Understanding Vector Quantization": https://huggingface.co/blog/ariG23498/understand-vq
- Keras docs (VQ-VAE example): https://keras.io/examples/generative/vq_vae/
- ML with a Honk: https://mlhonk.substack.com/p/14-vq-vae-vector-quantized-variational
- Shashank Yadav Medium: https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
- apxml.com VQ-VAE: https://apxml.com/courses/autoencoders-representation-learning/chapter-5-advanced-autoencoder-architectures/vector-quantized-vaes-vq-vae
- Lee Youngdo blog: https://leeyngdo.github.io/blog/generative-model/2023-09-02-VQ-VAE/
- `vector-quantize-pytorch` (lucidrains): https://github.com/lucidrains/vector-quantize-pytorch
- **FSQ** (대안 quantization): https://arxiv.org/abs/2309.15505
- **LFQ** (MagViT2): https://arxiv.org/abs/2310.05737

### Full-duplex Speech Dialogue Survey
- FD-SLM Survey (Chen & Yu 2025): https://arxiv.org/abs/2509.14515
- WavChat Survey: https://arxiv.org/pdf/2411.13577
- Recent Advances in Speech Language Models (Medium 요약): https://medium.com/@brijeshrn/from-turn-taking-to-synchronous-dialogue-building-and-measuring-true-full-duplex-systems-794a07f3e59f
- Full-Duplex-Bench: https://arxiv.org/html/2503.04721v1
- Full-Duplex-Bench GitHub: https://github.com/DanielLin94144/Full-Duplex-Bench
- Audio MultiChallenge: https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Audio_Multichallenge_Scale_LB.pdf

### True Full-duplex 모델
- **dGSLM** (Nguyen et al., Meta, 2023): Moshi 이전 유일한 full-duplex
- **SALMONN-omni**: https://arxiv.org/pdf/2505.17060
- **Hertz-dev** (Standard Intelligence, 2024): continuous latent 기반
- **PersonaPlex**: https://arxiv.org/html/2602.06053v1

### Pseudo FD / Engineered sync
- **DuplexCascade**: https://arxiv.org/pdf/2603.09180
- **Freeze-Omni** (Wang et al., 2024)
- **VITA-1.5**: dual LLM with shared KV cache
- **X-Talk**: https://arxiv.org/pdf/2512.18706

### Moshi 해설 자료
- Erogol paper review: https://erogol.substack.com/p/paper-review-moshi-a-speech-text
- Moonlight literature review: https://www.themoonlight.io/en/review/moshi-a-speech-text-foundation-model-for-real-time-dialogue
- Emergent Mind topic page: https://www.emergentmind.com/topics/moshi-a-speech-text-foundation-model
- ResearchGate paper (Algorithm 1 details): https://www.researchgate.net/publication/384563750_Moshi_a_speech-text_foundation_model_for_real-time_dialogue

### Inner Monologue 관련
- **Whisper** (word-level timestamps, Radford et al., 2023): https://arxiv.org/abs/2212.04356
- **AudioLM** (Borsos et al., 2022): https://arxiv.org/abs/2209.03143
- VAOS Voice Bridge gist (300자 burst injection 디버깅): https://gist.github.com/jmanhype/5aefd67d9e67b37a8b408abdab39b6d3
- Towards Japanese Full-duplex: https://arxiv.org/pdf/2506.02979

### Semantic Distillation
- **WavLM** (Chen et al., 2022): https://arxiv.org/abs/2110.13900
- **SpeechTokenizer** (Mimi가 차용): https://arxiv.org/abs/2308.16692

### RQ-Transformer 원형
- **Autoregressive Image Generation using Residual Quantization** (Lee et al., CVPR 2022): https://arxiv.org/abs/2203.01941

### Streaming Causal Conv & State Caching
- Balacoon, "Streaming Inference with Convolutional Layers": https://balacoon.com/blog/streaming_inference/
- Andrew Gibiansky, "Streaming Audio Synthesis": https://andrew.gibiansky.com/streaming-audio-synthesis/
- Stateful Conformer (Cache-based streaming): https://arxiv.org/html/2312.17279
- SpeechBrain Streaming Conformer Tutorial: https://speechbrain.readthedocs.io/en/latest/tutorials/nn/conformer-streaming-asr.html
- RAVE Streamable Neural Audio Synthesis: https://arxiv.org/pdf/2204.07064
- pytorch-tcn (실전 causal TCN 구현): https://github.com/paul-krug/pytorch-tcn

### Receptive Field 계산
- Luke Guerdan "Diving Into TCNs": https://lukeguerdan.com/blog/2019/intro-to-tcns/
- Unit8 TCN 가이드: https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
- Santi Pdp Medium "Receptive Fields in CNNs": https://medium.com/@santi.pdp/receptive-fields-in-convolutional-neural-networks-6368a699d838
- keras-tcn (RF 공식): https://github.com/philipperemy/keras-tcn

### 최적화 / 변형
- T-Mimi (decoder transformer-only, edge 최적화): https://arxiv.org/abs/2601.20094
- PersonaPlex 구현 노트: https://github.com/ivan-digital/qwen3-asr-swift/blob/main/docs/personaplex.md
- lucidrains `audiolm-pytorch`: https://github.com/lucidrains/audiolm-pytorch

---

## Aurchestra / Proactive Agent 연구 맥락에서의 시사점

이 노트의 내용을 정리하면서 발견한 연구상 적용 포인트:

### 1. Streaming TSE와 Mimi의 Buffer 구조 호환성

- TSE의 STFT-based 처리는 **overlap 필수** (75% overlap typical)
- Mimi/neural codec은 **stride 기반 non-overlapping**
- 두 paradigm을 같은 시스템에 붙일 때 STFT hop과 codec stride의 **공약수**를 frame 단위로 맞추는 게 깔끔
- 예: STFT hop 256, codec stride 1920 → LCM 단위로 sync

### 2. On-device 작은 모델 ↔ Server 큰 모델 구조

Moshi의 Inner Monologue가 자연스럽게 매핑되는 지점:

1. **on-device 작은 모델이 server의 큰 LLM 응답을 inner monologue처럼 frame 단위로 stream-in**
   — VAOS gist의 80ms 단위 20자 드립피드 패턴이 정확한 reference
2. **EPAD/PAD 같은 control token**으로 server 응답 도착 타이밍을 frame-aligned하게 신호
3. 학습 시 **α₂=100** 같은 weight 차등은 audio LM이 perceptual quality와 semantic correctness 사이
   trade-off를 명시적으로 다루는 좋은 선례

### 3. Mimi Encoder 단독 활용

24 kHz raw waveform을 12.5 Hz로 압축하는 streaming-friendly representation extractor로 활용 가능:
- TSE network 뒤에 붙여서 enhanced speech의 token-level representation 획득
- 그 위에 작은 on-device LLM을 올려 proactive cue 추론
- Hybrid 구조 가능성

### 4. Frame Rate의 Design Knob 의미

- Mimi가 12.5 Hz로 내려간 게 결정적 선택
- GPU에서 LLM이 12.5 step/sec (80ms budget per step)면 235B 같은 대형 모델도 따라잡을 수 있음
- 만약 75 Hz였으면 13ms budget이라 큰 모델은 불가능
- proactive agent에서 server-side 큰 모델 사용하려면 frame rate 설계가 핵심

### 5. 5종류의 Streaming State 관리

실전 시스템 구축 시 모든 cache를 일관성 있게 관리:
1. Mimi encoder conv state caches (≈220 ms)
2. Mimi encoder Transformer KV cache (~250–750 frames)
3. Mimi decoder conv state caches
4. Mimi decoder Transformer KV cache
5. RQ-Transformer KV cache

특히 interruption / barge-in 시 state reset 정책이 중요.

---

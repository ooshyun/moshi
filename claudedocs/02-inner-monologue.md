# Inner Monologue — 심층 분석

> 출처: Moshi 논문 §3.4.4 "Inner Monologue" + 본 저장소 소스코드 검증
> 핵심 코드: `lm.py`, `lm_utils.py`, `loaders.py`, `rust/moshi-core/src/lm_generate_multistream.rs`

---

## 0. ⚠️ 사전 주의 — "semantic"이라는 단어의 두 가지 용법

본 문서를 읽기 전에 명심:

| 용어 | 의미 | Vocab | 종류 |
|------|------|-------|------|
| **Mimi의 semantic codebook** (M0/U0) | RVQ 첫 코드북. WavLM distillation으로 의미적 audio token화 | 2048 | **AUDIO** |
| **Text token / Inner Monologue** (T) | SentencePiece subword. 실제 단어 | 32000 | **TEXT** |

둘 다 "semantic"이라고 불릴 수 있지만 **완전히 다른 stream**. Inner Monologue는 후자(텍스트)를 말함. 자세한 차이는 [`00 § 1.2.1`](00-moshi-architecture-overview.md#121-️-semantic-codebook은-텍스트-토큰이-아니다) 참조.

---

## 1. 개념 — 한 줄로

**Inner Monologue = Moshi가 자기 발화를 audio token으로 생성하기 직전에, 같은 발화의 텍스트 토큰을 먼저 생성하도록 강제하는 학습/추론 트릭.**

```
   ┌─────────────────────────────────────────────────────────────┐
   │  핵심 직관:                                                  │
   │                                                              │
   │  TTS = "텍스트 → 오디오"                                     │
   │  STT = "오디오 → 텍스트"                                     │
   │  Moshi = "텍스트와 오디오를 *동시에* 생성, 텍스트가 먼저"     │
   │                                                              │
   │  → 텍스트가 "내적 사고/계획"으로 작동                         │
   │  → audio token은 그 사고를 음성으로 "실현(realize)"           │
   └─────────────────────────────────────────────────────────────┘
```

---

## 2. 왜 필요한가? — Audio LM의 본질적 문제

### 2.1 순수 Audio LM의 한계

Mimi 같은 audio codec으로 음성을 토큰화해 LM을 학습하면 다음 문제가 발생함:

| 문제 | 설명 |
|------|------|
| **언어 모델링 부족** | RVQ token은 acoustic 정보가 강해서 *어휘/문법* 학습이 어려움 |
| **장기 일관성 부족** | "내가 무슨 말을 하고 있는지" 추적 불가능 → 횡설수설 |
| **사실성 부족** | 텍스트 코퍼스의 지식이 활용 안 됨 |
| **명료성 부족** | 발음은 자연스럽지만 내용은 의미 없는 경우 다수 발생 |

### 2.2 해결책 — 텍스트를 "내적 부산물"로 함께 생성

**Inner Monologue의 발상**: 사람도 말하기 전에 머릿속에서 단어를 떠올린다. 모델도 audio code를 만들기 전에 "이 시점에 어떤 단어를 말할까"를 텍스트로 먼저 출력하게 만들자.

→ 텍스트 LM의 풍부한 사전학습 지식을 audio 생성에 자연스럽게 주입.

---

## 3. 구조 — 텍스트 스트림이 어떻게 배치되는가

### 3.1 17개 스트림 중 텍스트는 1개

```
   Moshi LM의 입력/출력 코드북 배치  (lm.py:135-139, loaders.py:94-95)

   index:    0    1   2   3   4   5   6   7   8    9  10  11  12  13  14  15  16
            ┌──┬───────────────────────────────┬────────────────────────────────┐
   stream:  │T │ M0  M1  M2  M3  M4  M5  M6  M7│ U0  U1  U2  U3  U4  U5  U6  U7 │
            └──┴───────────────────────────────┴────────────────────────────────┘
             ▲           ▲                              ▲
             │           │                              │
       TEXT  │   MOSHI audio (8 codebooks)      USER audio (8 codebooks)
   (Inner    │   dep_q=8 (Depth가 생성)         n_q - dep_q = 8 (입력만)
   Monologue)│
             │
        audio_offset=1   (lm.py:299-304)
        num_codebooks = 1 (text) + 8 (Moshi) + 8 (user) = 17
```

**중요한 비대칭**:
- Moshi audio 8개는 **모델이 출력** (Depth Transformer가 생성, `lm.py:809`).
- User audio 8개는 **모델 입력만** (사용자 마이크에서 받음, `lm.py:683-696`).
- Text 1개는 **모델이 출력** (Temporal Transformer가 직접 생성, `lm.py:736`).

### 3.2 텍스트 토큰의 어휘

```
   ┌──────────────────────────────────────────────────────────────┐
   │  SentencePiece tokenizer (tokenizer_spm_32k_3.model)          │
   │  loaders.py:31, server.py:88                                  │
   │                                                                │
   │  vocab size = 32000  (text_card = 32000, loaders.py:92)       │
   │                                                                │
   │  특수 토큰:                                                    │
   │  ┌────┬─────────────┬───────────────────────────────────┐    │
   │  │ id │ 이름        │ 역할                              │    │
   │  ├────┼─────────────┼───────────────────────────────────┤    │
   │  │  0 │ <epad>      │ 한 단어의 "마지막 padding step"   │    │
   │  │    │             │ end_of_text_padding_id            │    │
   │  │    │             │ lm.py:100,260                     │    │
   │  │  1 │ <unk>       │ unknown                           │    │
   │  │  2 │ <eos>       │ end-of-sentence                   │    │
   │  │    │             │ run_inference.py:182              │    │
   │  │  3 │ <pad>       │ 단어 사이의 "침묵/이어짐" padding │    │
   │  │    │             │ existing_text_padding_id          │    │
   │  │    │             │ lm.py:99,256                      │    │
   │  │... │ "▁hello"등   │ SentencePiece subword            │    │
   │  │32000│ <start>    │ BOS, "initial token"              │    │
   │  │    │             │ text_initial_token_id             │    │
   │  │    │             │ lm.py:251                         │    │
   │  └────┴─────────────┴───────────────────────────────────┘    │
   └──────────────────────────────────────────────────────────────┘
```

---

## 4. 메커니즘 — 텍스트와 오디오의 시간 정렬

### 4.1 핵심 문제: rate mismatch

- **Audio frame rate**: 12.5 Hz (80 ms마다 한 토큰)
- **자연 speech text rate**: 평균 ~3-4 Hz (단어/초)
- 결과: **단어가 차지하는 audio frame 수가 다름** ("hello"는 ~5 frame, "I"는 ~1 frame).

### 4.2 해결: 단어 단위 padding alignment

학습 시 사용한 alignment 전략 (논문 §3.4.4):

```
   실제 음성 1초:    "  hello   world  "
                     │              │
                     ▼              ▼
   force-aligned:   t=0.10s "hello" 시작
                     t=0.32s "hello" 끝
                     t=0.50s "world" 시작
                     t=0.85s "world" 끝

   12.5Hz 프레임 인덱스: 0  1  2  3  4  5  6  7  8  9 10 11 12 ...
   (각 프레임=80ms)

   학습 시 텍스트 스트림 (Inner Monologue):
   ┌────────────────────────────────────────────────────────────┐
   │  t:    0       1     2      3      4      5      6     7  │
   │  text: [pad]→[hello]→[pad]→[pad]→[epad]→[world]→[pad]→... │
   │           3   "▁he"   3      3     0      "▁wo"   3        │
   │  audio:  -    🔊       🔊     🔊    🔊     🔊      🔊      │
   │      (silence) ────── "hello" 발음 ──── "world" 발음 ──    │
   └────────────────────────────────────────────────────────────┘

   규칙:
   1) 단어 시작 frame에서 word의 첫 token 출력
   2) 단어 발음 중인 frame들에서 word의 나머지 sub-token들 출력
   3) word 발음 끝 직전 frame에서 <epad> (id=0) 출력 — "다음 단어 곧 옴" 신호
   4) word 발음 끝부터 다음 word 시작까지 <pad> (id=3) 출력
```

### 4.3 추론 시 동작

추론 시에는 alignment 정보가 없으므로, **모델이 자동으로 적절한 시점에 word token을 sample**해야 함. 학습에서 익힌 패턴 덕분에 자연스럽게 일어남.

```
   서버에서 보이는 텍스트 토큰 시퀀스 (Frame 단위):
   ┌────────────────────────────────────────────────────────────────┐
   │ Frame:   0    1    2    3    4    5    6    7    8    9   10  │
   │ Token:   3    3    3    "Hi" 3    3    3    0   "▁there" 3  3 │
   │          ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑              │
   │        pad  pad  pad  "Hi" pad pad pad epad "there"            │
   │ Display: -    -    -    Hi   -    -    -    -   there           │
   │                                                                  │
   │ Audio:  𝅘𝅥𝅮  ─── "Hi" 발음 ───   ─── "there" 발음 ───            │
   │                                                                  │
   │ server.py:87:  if text_token not in (0, 3): show in UI          │
   │ → pad(3)와 epad(0)는 화면에 출력 안 함                          │
   └────────────────────────────────────────────────────────────────┘
```

---

## 5. 코드 흐름 — 한 step에서 무슨 일이 일어나나

### 5.1 다이어그램: Inner Monologue의 인과 순서

```
   ╔═══════════════════════════════════════════════════════════════════════════╗
   ║              Frame t — Inner Monologue Pipeline                            ║
   ║              ===================================                            ║
   ║                                                                              ║
   ║  Input cache at t-1 (17 streams):                                           ║
   ║  ┌──┬──────────────────────────────────┬───────────────────────────────┐  ║
   ║  │T │ M0 M1 M2 M3 M4 M5 M6 M7         │ U0 U1 U2 U3 U4 U5 U6 U7      │  ║
   ║  └──┴──────────────────────────────────┴───────────────────────────────┘  ║
   ║      │                │                              │                     ║
   ║      └────────────────┼──────────────────────────────┘                     ║
   ║                       │                                                     ║
   ║                       ▼                                                     ║
   ║              ┌──────────────────────┐                                       ║
   ║              │ Σ embeddings  (17×)  │   lm.py:390-397                       ║
   ║              │ → [B, 1, dim=4096]   │                                       ║
   ║              └──────────┬───────────┘                                       ║
   ║                         ▼                                                   ║
   ║              ┌──────────────────────────┐                                   ║
   ║              │  TEMPORAL TRANSFORMER    │   lm.py:402                       ║
   ║              │  32 layers, 7B params    │                                   ║
   ║              │  → transformer_out h_t   │                                   ║
   ║              └──────────┬───────────────┘                                   ║
   ║                         │                                                   ║
   ║              ┌──────────┴───────────┐                                       ║
   ║              ▼                      ▼                                       ║
   ║   ┌──────────────────┐    ┌──────────────────────┐                          ║
   ║   │ text_linear      │    │ depformer_in (8×)    │                          ║
   ║   │ h_t → V_text     │    │ h_t → 1024-d         │                          ║
   ║   └────────┬─────────┘    └──────────┬───────────┘                          ║
   ║            ▼                         │                                      ║
   ║   ┌──────────────────┐                │                                     ║
   ║   │ sample text_t    │   lm.py:736    │                                     ║
   ║   │   ⚠ TEXT FIRST   │                │                                     ║
   ║   └────────┬─────────┘                │                                     ║
   ║            │                          │                                     ║
   ║            └──────────┬───────────────┘                                     ║
   ║                       ▼                                                     ║
   ║   ╔═══════════════════════════════════════════════╗                          ║
   ║   ║   DEPTH TRANSFORMER (8 inner steps)            ║   lm.py:809-850         ║
   ║   ║                                                ║                          ║
   ║   ║   inner step 0:  input = text_t        ─► M0  ║   lm.py:818,433-434     ║
   ║   ║                                                ║                          ║
   ║   ║   inner step 1:  input = M0            ─► M1  ║                          ║
   ║   ║   inner step 2:  input = M1            ─► M2  ║                          ║
   ║   ║      ⋮                                         ║                          ║
   ║   ║   inner step 7:  input = M6            ─► M7  ║                          ║
   ║   ║                                                ║                          ║
   ║   ║   매 step마다 transformer_out h_t를 residual로  ║                          ║
   ║   ║   더해서 시간 컨텍스트 유지                     ║                          ║
   ║   ╚═══════════════════════════════════════════════╝                          ║
   ║                       │                                                     ║
   ║                       ▼                                                     ║
   ║   Output @ t:  text_t (Inner Monologue) + M0..M7 (Moshi audio)              ║
   ║                       │                                                     ║
   ║                       │  ⚠ 인과 순서가 중요:                                  ║
   ║                       │     1) 텍스트가 먼저 → 무엇을 말할지 결정             ║
   ║                       │     2) 그 텍스트를 조건으로 audio codes 생성         ║
   ║                       │                                                     ║
   ║                       ▼                                                     ║
   ║              cache.scatter(text_t, audio_t)  lm.py:762-772                  ║
   ║              (with per-codebook delays)                                     ║
   ╚═══════════════════════════════════════════════════════════════════════════╝
```

### 5.2 결정적 코드 라인 — "text가 먼저 출력된다"

```python
# lm.py:726-745  — Temporal에서 text_logits 추출 후 즉시 샘플
transformer_out, text_logits = state.graphed_main(input_, ...)
...
text_token = sample_token(
    text_logits.float(),
    self.use_sampling,
    self.temp_text,        # 별도 온도 (0.7), lm.py:563
    self.top_k_text,       # 별도 top-k (25), lm.py:564
)
text_token = text_token[:, 0, 0]   # shape [B]
if self.on_text_hook is not None:
    self.on_text_hook(text_token)   # ← 클라이언트에 즉시 푸시 가능

# lm.py:752  — Depth Transformer에 text_token 주입
audio_tokens = state.graphed_depth(text_token, transformer_out)
                                   ^^^^^^^^^^
                                   text가 Depth의 "seed" 역할
```

```python
# lm.py:809-841  — depformer_step (Depth Transformer 호출)
def depformer_step(self, text_token, transformer_out):
    prev_token = text_token              # ← TEXT가 첫 입력
    ...
    with lm_model.depformer.streaming(B_cfg):
        for cb_index in range(lm_model.dep_q):     # 8번 반복
            input_ = prev_token[:, None, None]
            logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
            next_token = sample_token(logits, ...)
            prev_token = next_token         # 다음 codebook의 input
            depformer_tokens.append(next_token)
    return torch.stack(depformer_tokens, dim=1)    # [B, 8]
```

```python
# lm.py:433-436  — Depth Transformer 내부: cb_index=0일 때 text embedding 사용
if cb_index == 0:
    token_in = self.depformer_text_emb(sequence[:, 0])
                ^^^^^^^^^^^^^^^^^^^^^^^^^
                ← 별도 text embedding (depformer_dim=1024)
else:
    token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + ...])
```

→ **Depth Transformer의 첫 입력이 text token임이 명백히 코드로 강제**되어 있음. 텍스트가 audio 생성의 직접적 조건.

---

## 6. Delay 구조와의 상호작용

Inner Monologue가 효과적이려면 **텍스트와 semantic audio token이 같은 시점에 정렬**되어야 함. 그래서 delay 설정이 다음과 같음:

```python
# loaders.py:118
delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
#         │  │  │                       │  │
#         │  │  │                       │  │
#       text M0 M1...M7              U0  U1...U7
#         │  │                          │
#         └──┴──◄ delay=0               └──◄ delay=0
#           (텍스트 ↔ Moshi semantic 정렬)  (사용자 audio도 semantic 정렬)
```

```
   Frame index:           t=0      t=1      t=2      t=3      ...
   ─────────────────────────────────────────────────────────────────
   text          (d=0)    text_0   text_1   text_2   text_3
   Moshi M0      (d=0)    M0_0     M0_1     M0_2     M0_3
   Moshi M1..M7  (d=1)    init     M1_0     M1_1     M1_2     ← 1 frame later
   User U0       (d=0)    U0_0     U0_1     U0_2     U0_3
   User U1..U7   (d=1)    init     U1_0     U1_1     U1_2     ← 1 frame later
   ─────────────────────────────────────────────────────────────────

   → text_t와 M0_t (Moshi semantic codebook)가 **동일 시점**에 생성.
   → semantic 코드북이 acoustic 코드북보다 1 frame 앞서감.
   → "텍스트 → semantic audio → acoustic detail"의 인과 사슬 형성.
```

이것이 **delayed streams modeling**과 inner monologue가 결합되는 방식.

---

## 7. 학습 vs 추론

### 7.1 학습 시 — Input vs Predicted의 결정적 비대칭

학습 시 한 프레임은 **(Moshi 측 audio + User 측 audio + Moshi 측 transcript)** 세 가지를 모두 ground truth로 가정합니다. 그러나 **17 streams 모두 입력**으로 들어가도 **Loss는 9 streams (text 1 + Moshi audio 8)에만** 걸립니다.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                  Training Asymmetry                                       │
   ├─────────────────────────────────────────────────────────────────────────┤
   │   Stream                  | Input? | Predicted? | Loss?                  │
   │   ────────────────────────┼────────┼────────────┼─────────               │
   │   Text (Inner Monologue)  |   ✅   |    ✅      |  ✅  CE                │
   │   Moshi M0..M7  (8개)     |   ✅   |    ✅      |  ✅  CE × 8            │
   │   User  U0..U7  (8개)     |   ✅   |    ❌      |  ❌  no loss           │
   │                                                                           │
   │   → User audio는 "관측만 한다" (모델이 user를 흉내내면 자기-대화로 망가짐)│
   │   → Moshi audio는 "생성한다"  (모델이 응답자 역할)                       │
   │   → Text는 "내적 사고를 노출한다" (Inner Monologue)                      │
   └─────────────────────────────────────────────────────────────────────────┘
```

코드 증거:

```python
# lm.py:40-46  — LMOutput에 text_logits와 audio logits가 함께 반환됨
@dataclass
class LMOutput:
    logits: torch.Tensor       # [B, K=8, T, card=2048]      Moshi audio logits
    mask: torch.Tensor
    text_logits: torch.Tensor  # [B, 1, T, text_card=32000]  TEXT logits ← 🔥
    text_mask: torch.Tensor

# lm.py:371-374  — Depth는 dep_q=8 stream에 대해서만 logits 생성
logits, logits_mask = _undelay_sequence(
    self.delays[self.audio_offset:self.audio_offset + self.dep_q],  # [1:9]
    logits, fill_value=float('NaN'))    # User stream(=index 9~16)은 제외!
```

학습 데이터 한 프레임의 구조:

```
   원본: 2명이 대화하는 stereo 오디오
       L 채널 = Moshi 측 음성 (응답자, AI가 흉내낼 사람)
       R 채널 = User 측 음성 (질문자)
       + 자막: Moshi 측 발화의 force-aligned transcript

   ↓ Mimi.encode + tokenize

   Frame t:
   ┌──┬─────────────────┬─────────────────┐
   │T │  M0 M1 ... M7   │  U0 U1 ... U7   │
   └──┴─────────────────┴─────────────────┘
    ↑       ↑                  ↑
    │       │                  └─ User 측 R채널을 Mimi로 토큰화
    │       └─ Moshi 측 L채널을 Mimi로 토큰화
    └─ Moshi 측 transcript의 t-시점 SentencePiece token (pad이면 3)
```

학습 시 teacher-forcing 인과 관계:

```
   입력 (t-1까지 ground truth 전부):
       T(t-1), M0..M7(t-1), U0..U7(t-1)
        ↓ Temporal Transformer

   예측:
      T_hat(t)    ──► CE vs T(t)     ✅ loss
        ↓ Depth Transformer (text(t) seed로 받음)
      M0_hat(t) ──► CE vs M0(t)      ✅ loss
      M1_hat(t) ──► CE vs M1(t)      ✅ loss
        ⋮
      M7_hat(t) ──► CE vs M7(t)      ✅ loss
      U0..U7(t) ← 다음 step의 입력으로만 사용 ❌ 예측 안 함
```

→ STT/TTS와의 결정적 차이:
- **STT**: audio → text (한 방향)
- **TTS**: text → audio (한 방향)
- **Moshi**: (user audio) → 동시에 (Moshi audio + Moshi text), **양방향 동시**

→ 학습 데이터셋의 구체적 구성에 대해서는 [`03-training-data.md`](03-training-data.md) 참조.

### 7.2 추론 시

```
   매 80ms마다:
   1) user.encode → user codes 8개 (캐시에 기록)
   2) Temporal forward → text logits
   3) Sample text token  ← "Moshi가 무슨 말 할까" 결정
   4) Depth forward (8 step) → Moshi audio codes 8개 ← 그 텍스트를 실현
   5) Mimi.decode(Moshi codes) → PCM → Opus → 클라이언트
   6) text token (id ∉ {0, 3}이면) → 클라이언트 자막
```

흥미로운 점: **자막이 음성보다 ~0~1 frame 빨리 도착함** (text delay=0, acoustic delay=1) → 클라이언트는 자막을 먼저 받고 음성이 따라 오는 자연스러운 UX.

---

## 8. 활용도 — Inner Monologue의 부수 효과

| 효과 | 어떻게 |
|------|--------|
| **무료 STT** | Moshi audio 출력 대신 user audio를 다른 모델에 입력 → text stream으로 자동 받아쓰기 (`STT` config, run_inference.py:121-127) |
| **무료 TTS** | text stream을 prefix로 강제 주입(prefill) → 그 텍스트의 음성이 생성됨 (`tts.py`, `lm_generate_multistream.rs`) |
| **자막 자연 생성** | 별도 ASR 없이 자막 표시 (server.py:86-92) |
| **사실성 향상** | 텍스트 LM처럼 사전학습된 지식 활용 → "Paris는 프랑스 수도" 같은 사실 응답 |
| **장기 일관성** | 텍스트 토큰이 짧기 때문에 context window 안에 더 긴 대화 이력 보관 가능 |
| **Hibiki(번역)** | 같은 구조로 입력 언어 음성 → 출력 언어 텍스트 + 음성 동시 생성 |

---

## 9. 참고 — TTS state machine (Inner Monologue를 거꾸로 이용)

TTS에서는 Moshi에게 **특정 텍스트를 말하게** 하기 위해 **inner monologue stream에 단어들을 강제 주입**함. Rust 구현이 이를 명확히 보여줌:

```rust
// rust/moshi-core/src/lm_generate_multistream.rs:152
if token_id == self.config.text_pad_token {        // 모델이 pad를 sample했다
    // 다음 단어를 sample할 자리. forced word를 주입.
}

// pad_mult로 pad token의 확률을 인위적으로 증가/감소시켜
// "지금 말을 시작해/멈춰" 제어 (line 252)
prs[self.config.text_pad_token as usize] *= f32::exp(*pad_mult);
```

→ Inner Monologue는 **추론 시 외부 제어 가능한 hook 지점**도 제공.

---

## 10. 한 줄 요약

> **Inner Monologue = "Moshi가 audio를 생성하기 직전에, 그 audio가 무슨 단어인지 텍스트로 먼저 출력하도록 강제하는 학습 트릭". 이를 통해 텍스트 LM의 지식·일관성·사실성을 audio 생성에 무료로 주입하고, 부수적으로 실시간 자막과 prompt 기반 TTS 제어도 얻는다.**

---

## 핵심 코드 진입점

| 경로 | 역할 |
|------|------|
| `moshi/models/lm.py:139` | text_emb 정의 (text vocab=32001) |
| `moshi/models/lm.py:141` | text_linear (text logits 출력) |
| `moshi/models/lm.py:736` | 추론 시 text token sampling (audio 이전) |
| `moshi/models/lm.py:818` | text_token을 Depth Transformer의 seed로 |
| `moshi/models/lm.py:433-434` | Depth의 cb=0일 때 depformer_text_emb 사용 |
| `moshi/models/loaders.py:99-100` | text_padding_id=3, end_padding_id=0 |
| `moshi/models/loaders.py:118` | delays — text와 semantic이 d=0으로 정렬 |
| `moshi/server.py:87` | UI 출력 시 pad(3)/epad(0) 필터링 |
| `rust/moshi-core/src/lm_generate_multistream.rs:18-46` | TTS용 inner monologue control config |

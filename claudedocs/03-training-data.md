# 학습/평가 데이터 구성

> 본 저장소는 **inference-only** 코드만 포함. 데이터셋과 학습 코드는 별도이며,
> 일부 fine-tuning 코드는 [`kyutai-labs/moshi-finetune`](https://github.com/kyutai-labs/moshi-finetune) 참조 (FAQ.md).
>
> 본 문서는 **Moshi 논문(arXiv:2410.00037) §5 Data + §6 Training**에서 발췌·요약했습니다.
> Kyutai는 사전학습 데이터셋을 **공개하지 않습니다** (FAQ: "We will not release the pre-training dataset").

---

## 1. 한눈에 보는 학습 단계

Moshi는 **6단계 커리큘럼**으로 학습됩니다. 각 단계가 요구하는 데이터가 다릅니다.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │  ① Helium (Text LM) 사전학습       2.1T tokens                          │
   │       │  text corpus (web, books)                                        │
   │       ▼                                                                  │
   │  ② Mimi (Audio Codec) 학습          7M hours unsupervised audio          │
   │       │  + WavLM distillation                                            │
   │       ▼                                                                  │
   │  ③ Moshi Audio Pretraining          7M hours, single-stream             │
   │       │  text + audio mixed, "speech as one stream"                      │
   │       ▼                                                                  │
   │  ④ Multi-stream Pretraining         100k hours stereo dialogue          │
   │       │  Moshi+User로 분리된 2-track 오디오                              │
   │       ▼                                                                  │
   │  ⑤ Instruct Fine-tuning             20k hours synthetic dialogue        │
   │       │  PyAnnote diarization + TTS로 합성                              │
   │       ▼                                                                  │
   │  ⑥ Voice / Speaker Fine-tuning      170 hours single-speaker TTS        │
   │          → Moshiko (male) / Moshika (female)                            │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘
```

각 단계를 차례로 봅시다.

---

## 2. ① Helium — Text-only LM 사전학습

**목적**: Moshi의 텍스트 지식·문법·상식을 갖춘 **7B text-only LM** 초기 가중치 확보.

| 항목 | 값 |
|------|----|
| 모델 이름 | Helium |
| 파라미터 | 7B |
| 학습 토큰 | **2.1 T tokens** |
| Architecture | Moshi의 Temporal Transformer와 동일 (32 layer, dim=4096) |
| 데이터 출처 | Wikipedia, StackExchange, OpenWebText2 등 일반 웹 + 책 + 학술논문 |
| 필터링 | CCNet style cleaning, dedup |

**의의**: 이 stage가 없었다면 inner monologue도 단순히 "alignment된 음소 시퀀스" 수준에 그쳤을 것. 사전학습된 텍스트 표현 덕분에 응답이 **사실적·일관적**.

코드와의 연결: `loaders.py:90` `_lm_kwargs`의 dim=4096, num_layers=32가 Helium 구조와 동일. Moshi는 Helium 가중치를 그대로 받아 audio modality를 얹어 확장 학습.

---

## 3. ② Mimi — Audio Codec 학습

**목적**: 24 kHz waveform → 12.5 Hz discrete codes로 토큰화하는 streaming codec.

### 3.1 데이터

| 항목 | 값 |
|------|----|
| 분량 | **약 7M hours** unsupervised audio |
| 도메인 | 음성·음악·환경음 혼합 |
| Resampling | 24 kHz mono |
| 라이선스 | 비공개 in-house collection |

### 3.2 학습 손실

```
   loss = adversarial (MS-STFT discriminator)
        + feature matching
        + commitment (RVQ)
        + WavLM distillation (first codebook only)
```

- ❌ **L1/L2 reconstruction loss 없음** — 적대적 학습만으로 perceptual 품질 추구.
- ✅ **WavLM distillation**: 첫 RVQ codebook이 WavLM 7번째 레이어 임베딩과 cosine 정렬 → semantic ↑.

### 3.3 평가

- **ViSQOL** (objective audio quality)
- **MOS** (subjective listening test) on internal data
- **Phoneme error rate** (semantic codebook의 의미 보존도)

---

## 4. ③ Moshi Audio Pretraining (단일 스트림)

**목적**: text를 audio token과 함께 모델링하는 능력 학습. **아직 dialogue 아님**.

### 4.1 데이터

- **약 7M hours** 단일 화자 발화 (대화 X)
- audio podcast / lecture / audiobook 등
- 자막은 **Whisper-large-v3로 자동 생성** (alignment까지)

### 4.2 입력 format

```
   이 단계에서는 17 stream이 아니라 9 stream만 사용:
   ┌──┬─────────────────┐
   │T │  M0 M1 ... M7   │      ← single speaker as "Moshi-only"
   └──┴─────────────────┘
       (User stream 없음)

   delays = [0, 0, 1, 1, 1, 1, 1, 1, 1]
```

### 4.3 핵심 — Inner Monologue Alignment

각 단어가 audio 어느 frame에서 시작되는지 정확하게 alignment 필요:

```
   1) Whisper-large-v3로 word-level timestamp 추출
   2) 12.5 Hz frame grid에 snap
   3) Moshi 측 transcript를 frame 단위로 expand:
        ┌──────────────────────────────────────────┐
        │ Frame:    0  1  2  3  4  5  6  7  8  9   │
        │ Token:    3  3  3 "Hi" 3  3  3  0 "▁there"│
        │           ↑  ↑  ↑   ↑                    │
        │         pad pad pad word                  │
        └──────────────────────────────────────────┘
   4) 단어 끝 직전 frame은 <epad>(0)으로 표시
```

→ 이 alignment가 **inner monologue의 모든 작동 원리의 학습 기반**.

---

## 5. ④ Multi-stream Pretraining (대화 학습)

**목적**: 진짜 full-duplex dialogue. Moshi와 User 두 화자를 분리된 stream으로 모델링.

### 5.1 데이터

| 항목 | 값 |
|------|----|
| 분량 | **약 100,000 hours** dialogue |
| 형식 | stereo audio (L=화자A, R=화자B) |
| 출처 | 다양한 multi-speaker corpus + 자체 수집 |
| 길이 | 평균 ~10분 대화 |

### 5.2 두 화자 분리 (Diarization)

원본이 mono인 경우 PyAnnote 등으로 화자 분리:

```
   ┌────────────────────────────────────────────────────────────┐
   │ 원본 mono audio (두 사람 대화)                              │
   │       ▼                                                     │
   │  PyAnnote Speaker Diarization                              │
   │       ▼                                                     │
   │  segment별 화자 라벨 (Speaker A / Speaker B)                │
   │       ▼                                                     │
   │  각 화자의 시간 구간을 mono track으로 분리                  │
   │   - A가 말할 때 → L channel, R channel은 silent             │
   │   - B가 말할 때 → R channel, L channel은 silent             │
   │       ▼                                                     │
   │  stereo audio with explicit speaker separation             │
   └────────────────────────────────────────────────────────────┘
```

문제: 한 화자 발화 중 다른 채널이 silent → "끼어들기/맞장구" 학습 불가.
해결: Section 5.3 (Fisher corpus).

### 5.3 Fisher Corpus — 진짜 stereo telephone dialogue

핵심 데이터:
- **Fisher English Phase 1+2**: 약 **2000 hours** 전화 대화
- 두 화자가 **각자 다른 마이크/채널**로 녹음 → 자연스럽게 stereo separated
- 끼어들기·맞장구·overlap이 자연 보존
- 8 kHz 원본 → 24 kHz로 upsample

```
   Fisher의 결정적 장점:
   - L 채널 = Alice의 mic (Bob 목소리는 약하게)
   - R 채널 = Bob의 mic   (Alice 목소리는 약하게)
   - 두 사람이 동시에 말해도 각자 채널에 독립 보존
   → Moshi가 "user가 말하는 도중에도 자기 채널에 reaction 토큰을 낸다"는
      full-duplex 행동을 학습할 수 있음
```

### 5.4 학습 데이터 입력 구조

```
   ┌────────────────────────────────────────────────────────────┐
   │ Stereo audio (10분, 24kHz, 2 channel)                       │
   │       ▼                                                     │
   │ L채널 → Mimi.encode → Moshi codes [B, 8, T_frames]          │
   │ R채널 → Mimi.encode → User  codes [B, 8, T_frames]          │
   │       ▼                                                     │
   │ L채널 transcript ← Whisper alignment → text [B, 1, T_frames]│
   │       ▼                                                     │
   │ Concat → input [B, 17, T_frames]                            │
   │       ▼                                                     │
   │ Apply delays + teacher forcing                              │
   └────────────────────────────────────────────────────────────┘
```

### 5.5 Speaker assignment — 누가 Moshi인가?

같은 데이터를 두 방향으로 사용:
- A를 Moshi(예측 대상), B를 User(조건)
- B를 Moshi(예측 대상), A를 User(조건)

→ 데이터 효율 2배. 모델은 "어느 화자든 응답할 수 있는" 일반화 능력 학습.

---

## 6. ⑤ Instruct Fine-tuning — 합성 대화

**목적**: Moshi를 단순 dialogue mimic이 아닌 **유용한 assistant**로 만들기.

### 6.1 문제

자연 dialogue 데이터는:
- 일상 잡담 위주 (질문답변/instruction 부족)
- 사실관계 답변 능력 학습 못 함
- "사용자: Paris 수도는? Moshi: ..."같은 instruction pattern 없음

### 6.2 합성 데이터 파이프라인

```
   ┌────────────────────────────────────────────────────────────┐
   │ Step 1: Helium (text LM)이 instruction-style dialogue 생성  │
   │   User: "Tell me about photosynthesis"                      │
   │   Moshi: "Photosynthesis is the process by which..."        │
   │                                                              │
   │ Step 2: 두 발화를 다른 TTS voice로 음성 합성                │
   │   - User → Voice A (다양한 voice pool에서 sampling)         │
   │   - Moshi → Voice B (Helium TTS, 다양함)                   │
   │                                                              │
   │ Step 3: 자연스러운 turn-taking timing 부여                  │
   │   - 응답까지 200~800ms pause                                │
   │   - 가끔 overlap                                            │
   │                                                              │
   │ Step 4: stereo로 mix (L=Moshi voice, R=User voice)         │
   │                                                              │
   │ Step 5: 학습 ④와 동일 format으로 입력                       │
   └────────────────────────────────────────────────────────────┘
```

### 6.3 분량

- 약 **20,000 hours** 합성 대화
- 다양한 topic: QA, casual chat, role play, technical Q&A
- Length variation: 1-turn (10초) ~ multi-turn (5분+)

### 6.4 보너스 — Voice diversity

Moshi 측 voice를 다양하게 만들어 학습 → **추론 시 다양한 voice를 흉내내는 능력** 확보. (Voice의 "feel"은 학습되지만 추론 시 control은 어려움 → 다음 단계로 보완.)

---

## 7. ⑥ Voice Fine-tuning — Moshiko / Moshika

**목적**: 안정적이고 일관된 **고정 voice** 확보. 일반 사용자에게 release할 수 있는 product 형태.

### 7.1 데이터

| 항목 | 값 |
|------|----|
| 분량 | **약 170 hours** |
| 화자 | 단일 TTS voice (Moshiko = male, Moshika = female) |
| 생성 방식 | Helium dialogue + TTS with the **single target voice** |
| 라이선스 | TTS voice는 인-house 라이선스 |

### 7.2 결과

- **Moshiko**: 남성 voice fine-tune
- **Moshika**: 여성 voice fine-tune
- HuggingFace에 공개: `kyutai/moshika-pytorch-bf16` 등 (loaders.py:35)

---

## 8. 평가 (Evaluation)

### 8.1 자동 평가

| 지표 | 무엇을 측정 | 데이터 |
|------|-------------|--------|
| **Perplexity** (text stream) | 언어 모델링 품질 | held-out dialogue transcripts |
| **Phoneme error rate** | semantic codebook의 의미 보존 | LibriSpeech-style test |
| **WER** (transcribe Moshi output) | Moshi 발화의 명료성 | Whisper로 다시 transcribe해서 비교 |
| **Streaming MOS proxy** | 응답 자연스러움 | DNSMOS/UTMOS |
| **Spoken QA accuracy** | 사실성 | TriviaQA / Web Questions를 음성으로 |

### 8.2 인간 평가

- **MOS** (5-scale audio quality)
- **Turn-taking naturalness** (얼마나 사람처럼 끼어들기/맞장구)
- **Helpfulness** (assistant로서 유용성)
- 비교 대상: OpenAI Voice mode, ChatGPT-4o, GPT-4o + TTS 파이프라인

### 8.3 Latency

- **End-to-end latency**: user 마지막 음성 + Moshi 첫 음성 사이 시간
- 측정 환경: L4 GPU, 동일 네트워크
- 보고치: ~200ms (README.md:40)

---

## 9. 핵심 통찰 — 데이터 측면

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  Moshi 데이터 설계의 핵심 아이디어                                   │
   │                                                                      │
   │  1. 텍스트 사전학습으로 "지식 + 일관성" 확보                          │
   │     → audio-only LM의 고질적 한계 해결                              │
   │                                                                      │
   │  2. Stereo telephone (Fisher) → 자연스러운 full-duplex 행동 학습     │
   │     → 끼어들기/맞장구/오버랩 모델링 가능                             │
   │                                                                      │
   │  3. Inner monologue alignment via Whisper                            │
   │     → 자동화 가능 (수작업 labeling 불필요)                          │
   │     → 사실상 무한 unlabeled audio를 학습 데이터로 변환               │
   │                                                                      │
   │  4. 합성 instruction dialogue                                        │
   │     → Helium text LM의 지식을 dialogue 형태로 audio domain에 주입    │
   │                                                                      │
   │  5. 마지막 단일 voice fine-tune                                      │
   │     → product-grade 일관성 확보                                      │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 10. 본 저장소에서 확인 가능한 단서

데이터셋 자체는 비공개이지만, 본 저장소에 남은 단서:

| 단서 | 위치 | 의미 |
|------|------|------|
| TTS voice 목록 | `data/tts.jsonl` | 합성 voice diversity 증거 |
| Hibiki sample | `data/sample_fr_hibiki_*.mp3` | 다국어 stream model 데모 |
| LM config | `loaders.py:90-119` | 학습 시 사용한 하이퍼파라미터 (delays 등) |
| text token IDs | `loaders.py:93`, `lm_generate_multistream.rs:30-31` | pad=3, epad=0의 정확한 정의 |
| `n_q=16`, `dep_q=8` | `loaders.py:94-95` | Multi-stream 17 codebook (text+Moshi+User) |
| Moshiko/Moshika weights | HF: `kyutai/moshiko-pytorch-bf16` | ⑥단계 결과물 |

---

## 11. 한 줄 요약

> **Moshi는 (1) 텍스트 LM 사전학습 → (2) Mimi codec 학습 → (3) 단일화자 audio+text → (4) Fisher 등 stereo dialogue → (5) 합성 instruct dialogue → (6) 단일 voice fine-tune**의 6단계 커리큘럼으로 학습됩니다. 데이터셋은 비공개이지만 자동화 가능한 alignment(Whisper) + 합성(Helium TTS)으로 대부분 unlabeled audio에서 ground truth를 만들어내는 것이 핵심 트릭입니다.

---

## 12. 참고

- 논문 §5 Data, §6 Training: <https://arxiv.org/abs/2410.00037>
- 공식 finetune 코드: <https://github.com/kyutai-labs/moshi-finetune>
- Fisher Corpus: LDC2004T19, LDC2005T19
- WavLM (Mimi distillation source): <https://arxiv.org/abs/2110.13900>
- Whisper (alignment): <https://github.com/openai/whisper>
- PyAnnote (diarization): <https://github.com/pyannote/pyannote-audio>

# Opus Decoder — Moshi에서의 역할

## 1. Opus 코덱이란?

**Opus**는 IETF 표준(RFC 6716, 2012)으로 정의된 **로열티프리 lossy 오디오 코덱**. WebRTC, Zoom, Discord, YouTube Live 등 거의 모든 실시간 오디오 시스템의 사실상 표준.

### 1.1 핵심 특징

| 속성 | 값 | 비고 |
|------|----|----|
| 비트레이트 | 6 kbps ~ 510 kbps | 동적 적응 가능 |
| 알고리즘 지연 | **2.5 ms ~ 60 ms** (보통 20 ms) | 실시간 통신 가능 |
| 샘플링레이트 | 8/12/16/24/48 kHz | full-band까지 지원 |
| 채널 | mono / stereo | |
| 라이선스 | BSD 3-clause, 특허 free | 상업 사용 자유 |
| 표준 라이브러리 | `libopus` (Xiph.Org) | C로 구현 |

### 1.2 내부 구조 — Hybrid Codec

Opus는 **두 코덱을 동적으로 결합**한 하이브리드 구조:

```
            ┌─────────────────────────────────────────────┐
            │              OPUS ENCODER                    │
            │                                              │
   PCM ─────┤   ┌───────────────┐    ┌───────────────┐    ├──── opus bitstream
            │   │  SILK         │    │  CELT         │    │
            │   │  (음성용)      │    │  (음악/광대역) │    │
            │   │  Linear       │    │  MDCT + range │    │
            │   │  Prediction   │    │  coder        │    │
            │   │  Skype 출처   │    │  Xiph 출처    │    │
            │   └───────────────┘    └───────────────┘    │
            │            \                /                │
            │             \              /                 │
            │           Hybrid mode (낮은 주파수=SILK,      │
            │                       높은 주파수=CELT)       │
            └─────────────────────────────────────────────┘
```

- **SILK**: 음성에 최적 (Skype에서 기증). 8~12 kHz band에 우수.
- **CELT**: 음악/광대역에 최적. MDCT + range coding.
- 인코더가 콘텐츠를 보고 SILK/CELT/Hybrid 모드를 매 프레임 자동 선택.

---

## 2. Moshi에서의 Opus 사용

### 2.1 데이터 경로

```
   Browser microphone (Web Audio API)
         │
         │  PCM 24kHz mono Float32, frame=80ms (1920 samples)
         ▼
   ┌─────────────────────────────────┐
   │  Browser-side Opus encoder      │   client/src/audio-processor.ts
   │  (MediaRecorder or WebCodecs)   │
   └─────────────────────────────────┘
         │
         │  binary Opus packets (수십~수백 bytes)
         ▼
   ┌─────────────────────────────────┐
   │  WebSocket frame:               │
   │     b"\x01" + opus_bytes        │   1 = audio tag
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │  Server: sphn.OpusStreamReader  │   server.py:122
   │  .append_bytes(opus_bytes)      │   sphn = Rust libopus wrapper
   │  → PCM 24kHz Float32 numpy      │   pip install sphn
   └─────────────────────────────────┘
         │
         ▼
   accumulate until ≥1920 samples (80ms)
         │
         ▼
   torch.Tensor[1, 1, 1920]  →  Mimi.encode()  →  user codes [1, 8, 1]
```

### 2.2 핵심 코드 (server.py)

```python
# server.py:161 — Opus 스트리밍 객체 생성 (24kHz)
opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

# server.py:119-135 — 수신: opus_bytes → PCM
if kind == 1:  # audio tag
    payload = message[1:]
    pcm = opus_reader.append_bytes(payload)   # ← libopus 호출
    if all_pcm_data is None:
        all_pcm_data = pcm
    else:
        all_pcm_data = np.concatenate((all_pcm_data, pcm))
    while all_pcm_data.shape[-1] >= self.frame_size:    # frame_size=1920
        chunk = all_pcm_data[:self.frame_size]
        chunk = torch.from_numpy(chunk).to(self.device)[None, None]
        codes = self.mimi.encode(chunk)   # PCM → Mimi codes
        ...

# server.py:82-85 — 송신: Mimi.decode 출력 PCM → Opus
main_pcm = self.mimi.decode(tokens[:, 1:])  # Moshi가 만든 PCM
main_pcm = main_pcm.cpu()
opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())   # ← libopus
if len(opus_bytes) > 0:
    await ws.send_bytes(b"\x01" + opus_bytes)
```

### 2.3 sphn 라이브러리

`sphn`은 Kyutai에서 만든 Rust 기반 Python 오디오 I/O 라이브러리 (`rust/sphn` GitHub repo 별도). 내부적으로 **`audiopus` (libopus의 Rust 바인딩) + 자체 ring buffer**를 사용해 stream-friendly Opus 처리 제공.

- `OpusStreamReader`: 부분 opus packet을 받아도 buffering 후 디코딩
- `OpusStreamWriter`: 들어오는 PCM을 누적했다가 충분히 모이면 opus packet 출력

---

## 3. **왜** Opus를 쓰는가? — 설계 의도

### 3.1 대안과의 비교

| 옵션 | 대역폭 (24kHz mono) | 지연 | 비고 |
|------|---------------------|------|------|
| **Raw PCM 16-bit** | 384 kbps | 0 ms | 네트워크 부하 매우 큼 |
| Raw PCM 32-bit float | 768 kbps | 0 ms | 더 큼 |
| MP3 | 32~128 kbps | ~150 ms | 지연이 큼, 음성 비최적 |
| AAC-LD | 24~64 kbps | ~20 ms | 라이선스 이슈 |
| **Opus** | **24~32 kbps** | **20 ms** | **표준, 무료, 브라우저 내장** |
| Mimi codes | ~1.1 kbps | 80 ms (Mimi 자체) | Moshi의 내부 표현 |

### 3.2 결정의 핵심

```
   ┌──────────────────────────────────────────────────────────┐
   │  Q: 그냥 Mimi codes를 직접 전송하면 안 되나?               │
   │                                                            │
   │  A: 안 됨. 이유:                                           │
   │  1) 브라우저가 Mimi를 모름 → 클라이언트에 Mimi 실행 필요   │
   │  2) Mimi는 GPU/연산 자원이 필요 (작은 디바이스 부적합)     │
   │  3) Opus는 모든 브라우저에 하드웨어 가속 내장              │
   │  4) Opus(24 kbps)는 Mimi(1.1 kbps)보다 크지만 무시 가능    │
   │     수준. 진짜 병목은 모델 추론.                            │
   └──────────────────────────────────────────────────────────┘
```

### 3.3 지연(latency) 관점

Opus는 Moshi의 200 ms 전체 latency 중 **20 ms 정도만 차지** — Mimi의 80 ms 프레임 지연이 훨씬 큼. 따라서 Opus는 latency 병목이 아님.

```
   Total ~200 ms latency budget:
   ┌────────┬────────┬────────┬──────────┬────────┬────────┐
   │ Mic    │ Opus   │ Net    │ Mimi enc │ LM     │ Acoustic│
   │ buffer │ encode │ RTT    │ + LM     │ Depth  │ delay   │
   │ 80 ms  │ 20 ms  │ <20 ms │  30 ms   │  30 ms │  80 ms  │
   └────────┴────────┴────────┴──────────┴────────┴────────┘
                  Opus는 작은 부분에 불과
```

---

## 4. 자주 묻는 질문

### Q1. 다이어그램의 "Opus Decoder (sphn)"는 정확히 무엇인가?

> 서버에서 **클라이언트가 보낸 Opus 패킷을 PCM으로 풀어내는 디코더**. `sphn.OpusStreamReader`가 wrapper이고, 내부는 `libopus`. 24 kHz PCM Float32로 풀려서 `numpy.ndarray`로 반환됨. 80 ms(1920 sample)가 모일 때까지 ring buffer에 누적했다가 Mimi encoder에 입력됨.

### Q2. 왜 PCM 자체를 보내지 않나?

> 24 kHz × 16-bit = **384 kbps**. WebSocket으로 보내기엔 부담스럽고, 모바일 네트워크에서 패킷 손실 시 ringing/glitch가 발생함. Opus는 **packet loss concealment** (PLC)도 지원해서 1~2개 packet 손실 정도는 자연스럽게 매꿔줌.

### Q3. Opus와 Mimi는 어떻게 다른가?

> 둘 다 audio codec이지만 **목적이 다름**:
> - **Opus**: 사람 귀로 듣기 위한 압축. 결과는 PCM(또는 Opus packet).
> - **Mimi**: LM이 다루기 위한 **이산 토큰화**. 결과는 RVQ 정수 코드 (`[B, K=8, T]`).
>
> 다이어그램에서:
> ```
>   Browser ── Opus ──▶ Server PCM ──▶ Mimi tokens ──▶ LM ──▶ Mimi tokens ──▶ PCM ──▶ Opus ──▶ Browser
>            (네트워크 전송용)        (모델 입력용)              (모델 출력)    (네트워크 전송용)
> ```

### Q4. 클라이언트 측은 어떻게 디코딩하나?

> 브라우저의 `AudioDecoder` (WebCodecs API) 또는 자체 Opus WASM 디코더 사용. 코드: `client/src/decoder/`. 디코딩된 PCM은 `AudioWorklet`을 거쳐 스피커로 출력.

### Q5. SSL/HTTPS는 왜 필요한가?

> Opus와 무관함. **브라우저 정책 때문**: HTTP origin은 마이크 권한을 받을 수 없음 (보안). 로컬호스트만 예외. 따라서 원격 서버는 반드시 HTTPS여야 마이크 접근 가능 → `server.py:265-272`에서 ssl_context 설정.

---

## 5. 참고

- RFC 6716: <https://datatracker.ietf.org/doc/html/rfc6716>
- Opus 공식 사이트: <https://opus-codec.org/>
- sphn (Kyutai): <https://github.com/kyutai-labs/sphn>
- 코드 진입점: `moshi/server.py:122` (서버 디코딩), `moshi/server.py:83` (서버 인코딩)

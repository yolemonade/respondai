---

# RespondAI

## 프로젝트 기본 정보

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | RespondAI (Call-and-Response Improv Game) |
| 과목 | Deep Learning for Music and Audio |
| 팀 구성 | 2인 팀 |
| 개발 기간 | 4주 |
| 확정일 | 2026-05-10 |
| 상태 | 주제 확정 / 데이터셋 검증 완료 |

---

## 프로젝트 개요

### 한 줄 요약

> 사용자가 가상 피아노 또는 마이크 허밍으로 4마디를 입력하면, 직접 학습한 Transformer 모델이 음악적으로 연관된 4마디로 응답하는 AI 노래 즉흥 배틀 게임.

### 배경 및 동기

즉흥 연주(Improvisation)는 재즈 등 다양한 음악 장르에서 핵심적인 표현 방식으로, 연주자들이 서로의 멜로디에 반응하며 음악적 대화를 나누는 Call-and-Response 형식이 오랫동안 활용되어 왔다. 본 프로젝트는 이 구조를 딥러닝 기반 심볼릭 음악 생성 모델로 구현하여, 음악적 지식이 없는 일반 사용자도 가상 피아노 클릭이나 허밍만으로 AI와 즉흥 음악 배틀을 즐길 수 있는 인터랙티브 게임을 만드는 것을 목표로 한다.

### 강의 연관성

| 강의 챕터 | 연관 내용 |
| --- | --- |
| 2강 — 소리와 디지털 오디오 | 허밍 오디오 신호 처리, 샘플링, 피치 추출 (허밍 모드) |
| 3강 — 딥러닝을 이용한 음악 오디오 분류 | PESTO ViT 기반 피치 검출 모델 구조 이해 (허밍 모드) |
| 6강 — Symbolic Music Generation | **핵심 연관** — 음악을 Language Model로 다루기, auto-regressive 생성, muspy 활용 |

---

## 시스템 아키텍처

### 전체 파이프라인

입력 방식에 따라 두 가지 모드로 분기된다. 공통 파이프라인(모티프 분석 이후)은 동일하다.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [모드 선택]
   ┌────────────────────┐         ┌──────────────────────┐
   │  🎹 피아노 모드     │         │  🎤 허밍 모드 (Beta)  │
   │  (기본, 권장)       │         │  (보조, 정확도 제한)  │
   └────────────────────┘         └──────────────────────┘
            ↓                               ↓
[가상 피아노 / 키보드 클릭 입력]    [마이크 입력 — 사용자 허밍 4마디]
  - 클릭으로 음표 직접 입력                  ↓
  - 실시간 피아노롤 표시          [오디오 전처리 — librosa]
  - note sequence 즉시 생성        - 샘플링 레이트: 16kHz
            ↓                      - 무음 구간 필터링
            │                               ↓
            │                  [피치 검출 — PESTO (ViT 기반, pretrained)]
            │                    - 실시간 스트리밍 지원
            │                    - 연속 pitch contour 추출
            │                    - confidence threshold: 0.8 이상만 사용
            │                               ↓
            │                  [MIDI 변환 & 양자화 — pretty_midi]
            │                    - median filter로 pitch contour 스무딩
            │                    - onset detection으로 음 경계 검출
            │                    - 각 구간 중위 피치 → MIDI note 변환
            │                    - 16분음표 그리드에 nearest-neighbor snap
            │                    - 옥타브 점프 보정 (2옥타브 이상 이동 시)
            │                    ※ 양자화 정확도 제한 있음 (아래 평가 지표 참고)
            │                               ↓
━━━━━━━━━━━━━━━━━━━━ 공통 파이프라인 ━━━━━━━━━━━━━━━━━━━━
            ↓
[모티프·화성 분석 — music21]
  - 조성(Key) 추출
  - 주요 음정 간격 패턴 분석
  - 리듬 패턴 추출
            ↓
[Call-and-Response Transformer — 직접 구현 및 학습]
  - [CALL] 토큰으로 사용자 멜로디 prefix
  - [RESPONSE] 이후 auto-regressive 생성
            ↓
[MIDI 후처리]
  - 생성된 토큰 → MIDI note sequence 디코딩
  - 조성 보정 (CALL과 동일 키 유지)
            ↓
[오디오 렌더링 — FluidSynth + soundfont]
            ↓
[게임 UI 출력]
  - 피아노롤 시각화
  - 점수 계산
  - 다음 라운드 진행
```

### 입력 전략 — 2모드 구성

| 모드 | 입력 방식 | 상태 | 정확도 | 비고 |
| --- | --- | --- | --- | --- |
| 🎹 피아노 모드 (기본) | 가상 피아노 클릭 또는 키보드 입력 | ✅ 메인 기능, 데모 보장 | 높음 (직접 입력) | 기본 모드, 발표 데모 사용 |
| 🎤 허밍 모드 (Beta) | 실시간 마이크 허밍 → PESTO | ⚠️ 보조 기능, 정확도 제한 | 제한적 (양자화 어려움) | 도전적 시도, 오류율 명시 |

> **설계 원칙:** 허밍에서 연속 f0를 추출하는 것은 상대적으로 용이하나, 이를 이산 음표(discrete note)로 양자화하는 과정은 음 경계 검출, 비브라토 처리, 리듬 그리드 스냅 등 복합적인 어려움이 따른다. 피아노 모드를 기본으로 안정적인 데모를 보장하되, 허밍 모드는 도전적 기능으로 병행 제공한다.

---

## 핵심 모델 명세 — Call-and-Response Transformer

### 모델 개요

GPT 스타일의 Decoder-only Transformer를 PyTorch로 처음부터 직접 구현하고 학습한다. 외부 Transformer 라이브러리(HuggingFace 등)를 가져다 쓰지 않으며, `nn.Module`을 직접 상속해 Self-Attention, Causal Mask, Positional Encoding을 구현한다.

### 아키텍처 상세

| 구성 요소 | Sanity Check 설정 | 본 학습 설정 |
| --- | --- | --- |
| 레이어 수 | 4 | 6 |
| 어텐션 헤드 | 4 | 8 |
| 임베딩 차원 (d_model) | 256 | 512 |
| FFN 차원 (d_ff) | 512 | 2048 |
| 최대 시퀀스 길이 | 512 | 512 |
| Dropout | 0.1 | 0.1 |
| 총 파라미터 수 | ~3M | ~10M |
| Attention 타입 | Causal Self-Attention | Causal Self-Attention |

### 토크나이저 설계

**어휘(Vocabulary) 구성:**

| 토큰 종류 | 범위 | 설명 |
| --- | --- | --- |
| 특수 토큰 | 0~4 | `[PAD]`, `[CALL]`, `[SEP]`, `[RESPONSE]`, `[EOS]` |
| pitch 토큰 | 5~132 | MIDI pitch 0~127 (총 128개) |
| duration 토큰 | 133~164 | 16분음표 단위 1~32 (총 32개) |
| rest 토큰 | 165~196 | 16분음표 단위 쉼표 1~32 (총 32개) |
| 전체 vocab size | **197** | |

**시퀀스 구조:**

```
[CALL] pitch:62 dur:4 pitch:64 dur:2 rest:2 pitch:65 dur:4 pitch:62 dur:2
pitch:64 dur:4 pitch:65 dur:2 pitch:67 dur:4 rest:2
[SEP]
[RESPONSE] pitch:65 dur:4 pitch:67 dur:2 pitch:69 dur:4 pitch:67 dur:2
pitch:65 dur:4 pitch:64 dur:2 pitch:62 dur:4 rest:2
[EOS]
```

### 학습 방식

- **목적 함수:** Cross-Entropy Loss (다음 토큰 예측, auto-regressive)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.01)
- **스케줄러:** Warmup + Cosine Annealing (warmup_steps=500)
- **Causal Mask:** 미래 토큰을 참조하지 못하도록 하삼각 마스크 적용
- **컨디셔닝:** `[CALL]` ~ `[SEP]` 구간을 prefix로 고정하고 `[RESPONSE]` 이후만 loss 계산

---

## 데이터셋 명세

### 검증 완료 현황 (2026-05-10 기준)

| 데이터셋 | 곡 수 | 학습 pairs | 로컬 사용 가능 | 용도 |
| --- | --- | --- | --- | --- |
| Nottingham Database | 1,034곡 | ~4,136 | ✅ 즉시 가능 | sanity check + 보조 학습 |
| JSB Chorales | 371곡 | ~5,900 | ✅ 즉시 가능 | sanity check |
| Lakh MIDI Dataset | ~17,000곡 | ~100,000+ | ⚠️ Colab 필요 | 본 학습 메인 데이터 |

### Call-and-Response 쌍 생성 전략

```
슬라이딩 윈도우 (stride=2):
  - bars 1~4  →  [CALL]
  - bars 5~8  →  [RESPONSE]
  - bars 3~6  →  [CALL]
  - bars 7~10 →  [RESPONSE]
  - ...

조건:
  - 4마디 미만 곡 스킵
  - CALL과 RESPONSE가 같은 조성(key signature)일 것
  - 파싱 에러 파일 자동 스킵
```

### 데이터 전처리 파이프라인

```
MIDI 파일 로드 (pretty_midi)
      ↓
단선율(monophonic) 트랙 추출
      ↓
note sequence 변환 (pitch, duration, velocity)
      ↓
16분음표 단위 양자화
      ↓
조성 분석 (music21)
      ↓
[CALL] / [RESPONSE] 슬라이딩 윈도우 적용
      ↓
토크나이저로 정수 시퀀스 변환
      ↓
최대 길이 512 토큰으로 패딩/트런케이션
```

---

## 평가 지표

### 정량 지표 (모델)

| 지표 | 설명 | 목표값 |
| --- | --- | --- |
| Perplexity | 언어 모델 기본 지표, 낮을수록 좋음 | < 10 (LMD 학습 기준) |
| 조성 일관성 | CALL과 RESPONSE의 키 일치율 | > 80% |
| 리듬 유사도 | 박자 패턴 피어슨 상관계수 | > 0.5 |
| 모티프 활용률 | CALL 핵심 음형이 RESPONSE에 등장하는 비율 | > 30% |
| 추론 속도 | RESPONSE 4마디 생성 소요 시간 | < 2초 |

### 정량 지표 (입력 파이프라인)

| 지표 | 설명 | 목표값 | 적용 모드 |
| --- | --- | --- | --- |
| 피치 검출 정확도 | PESTO 허밍 → MIDI 변환 음정 오류율 | < 35% ⚠️ | 허밍 모드 |
| 양자화 정확도 | 리듬 양자화 후 박자 오차 | ±2 16분음표 이내 ⚠️ | 허밍 모드 |
| 피아노 입력 지연 | 클릭 → 피아노롤 반영 소요 시간 | < 50ms | 피아노 모드 |

> ⚠️ **허밍 모드 정확도 한계 명시:** 연속 f0를 이산 음표로 양자화하는 과정(음 경계 검출, 비브라토 처리, 리듬 그리드 스냅)은 기술적으로 난이도가 높아 오류율이 높을 수 있다. 허밍 모드는 도전적 기능으로 제공하며, 발표 시 이 한계를 명시한다.

### 정성 평가

- 팀원 청취 평가 (5점 척도): AI 응답이 음악적으로 자연스러운가
- 발표 현장 청중 평가: 게임 데모 플레이 후 재미·완성도 평가

---

## 프로젝트 타당성 검토

### 딥러닝과의 연관성

딥러닝 요소가 두 층으로 구성된다.

**① 핵심 모델 — Call-and-Response Transformer (직접 구현·학습)**
Decoder-only Transformer의 Causal Self-Attention, Positional Encoding, Auto-regressive 생성을 PyTorch로 처음부터 구현하고 MIDI 데이터로 학습한다. 강의 6강의 Symbolic Music Generation 내용(음악을 Language Model로 다루기, auto-regressive 생성, muspy 활용)이 그대로 적용된다.

**② 입력 파이프라인 — PESTO 피치 검출 (ViT 기반 딥러닝, 허밍 모드)**
허밍 오디오에서 pitch를 추출하는 PESTO는 ViT(Vision Transformer) 기반 딥러닝 모델로, 실시간 스트리밍을 기본 지원하며 강의 2~3강의 오디오 신호 처리·딥러닝 분류 내용과 연결된다. pretrained 모델을 활용하지만, 전체 파이프라인에서 딥러닝 모델 두 가지가 직렬로 연결되는 구조다.

### 직접 구현한 모델인가

핵심 생성 모델(Transformer)은 MusicGen, HuggingFace 등 외부 라이브러리를 사용하지 않고, PyTorch `nn.Module`로 어텐션 메커니즘부터 직접 설계·구현·학습한다. 파라미터 수, 레이어 구조, 컨디셔닝 전략, 토크나이저 설계까지 모두 직접 결정하며, 대학 최종 프로젝트로서 "직접 구현한 모델" 요건을 충족한다.

### 기존 연구와의 차별점

Google Magenta의 AI Duet, Anticipatory Music Transformer 등 유사 연구가 존재하지만, 본 프로젝트는 세 가지 측면에서 차별화된다.

1. **멀티모드 인터페이스** — 가상 피아노(기본)와 허밍(Beta) 두 가지 입력 방식으로 다양한 사용자 접근성 확보
2. **게임 배틀 형식** — 단순 생성 데모가 아닌 5라운드 배틀 구조로 인터랙티브 경험 제공
3. **음악적 품질 정량화** — 모티프·화성·리듬 분석을 점수 체계로 연결해 AI 응답 품질을 시각적으로 피드백

기존 연구를 재현하는 것에 그치지 않고, 직접 학습한 모델 위에 독자적인 게임 경험을 설계한 것으로 최종 프로젝트로서 충분한 의의가 있다.

### 데이터셋 가용성

| 데이터셋 | 상태 |
| --- | --- |
| Nottingham DB (1,034곡) | ✅ 로컬 검증 완료, 즉시 사용 가능 |
| JSB Chorales (371곡) | ✅ 로컬 검증 완료, 즉시 사용 가능 |
| Lakh MIDI Dataset (~17,000곡) | ✅ 공개 데이터, Colab에서 다운로드 확인됨 |

세 데이터셋 모두 무료 공개 데이터로 라이선스 문제 없으며, 합산 학습 pairs ~110,000개 이상 확보 가능하다.

---

## 참고 자료

- Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/
- muspy 라이브러리: https://salu133445.github.io/muspy/
- PESTO 피치 검출 논문 및 구현체: https://github.com/SonyCSLParis/pesto
- pretty_midi: https://craffel.github.io/pretty-midi/
- music21: https://web.mit.edu/music21/
- Anticipatory Music Transformer (참고 논문): https://arxiv.org/abs/2306.08620
- Attention Is All You Need (Transformer 원논문): https://arxiv.org/abs/1706.03762
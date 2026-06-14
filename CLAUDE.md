# RespondAI — Team B (강유영)

## 절대 규칙
- `data/` `model/` `inference/` `analysis/` `training/` — **A팀 코드, 수정 금지**
- 게임 전체 규칙·화면 구성·점수 체계는 **`prompt.md`** 가 유일한 기준

## A팀 임포트 인터페이스
```python
from data.tokenizer import Note, KEY_NAMES, TEMPO_BINS
# Note(pitch, start, end) — 시간 단위: 16분음표 스텝 (정수)

from inference.generate import generate, load_model_for_inference
# Phase 2부터 사용. generate() → GenerationResult (튜플 아님)
# result.response_notes  result.attn_scores  result.stop_reason

from inference.decode import notes_to_wav

from analysis.scoring import score_response, session_summary
# score_response() → {"key_consistency", "rhythm_similarity", "motif_usage",
#                      "creativity_bonus", "total", "feedback"}
# session_summary(round_results) → {"grade", "total", "base_score", "bonus_score"}

from input.piano import mock_ai_response   # Phase 1 전용
```

## 주의사항
- R5 모티프 보너스 +150 → `session_summary()` 미포함, app에서 직접 더할 것
- A팀 API 상세 스펙: `docs/PLAN.md`
- 허밍 모드 구현: `humming` 브랜치

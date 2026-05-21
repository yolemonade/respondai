# RespondAI — A팀 API 참조 (Team B용)

## 모델 연동 (Phase 2)

```python
# 앱 시작 시 1회 로드
model, tokenizer, device = load_model_for_inference("checkpoints/full/best.pt")

# AI 응답 생성
result = generate(model, tokenizer, call_notes, key="Dm", tempo=92, max_bars=4)
# GenerationResult dataclass — 튜플 아님
# result.response_notes  : List[Note]
# result.attn_scores     : attention weights (VIZ-02용)
# result.stop_reason     : str

# 교환별 점수
score = score_response(call_notes, result.response_notes, key="Dm")
# {"key_consistency": int, "rhythm_similarity": int, "motif_usage": int,
#  "creativity_bonus": int, "total": int, "feedback": str}

# 최종 세션 집계
summary = session_summary(round_results)
# {"grade": "S"/"A"/"B"/"C", "total": int, "base_score": int, "bonus_score": int}
# ※ R5 모티프 보너스 +150은 session_summary에 포함 안 됨 → 직접 더할 것

# 오디오 렌더링
audio_path = notes_to_wav(notes, tempo=92, sound_font="/path/to/file.sf2")
```

## Phase 구분

| Phase | 조건 | mock → 실제 전환 |
|---|---|---|
| Phase 1 | A 모델 없이 UI 완성 | `mock_ai_response()` 사용 |
| Phase 2 | A sanity 체크포인트 완성 후 | `generate()` 로 교체 |
| Phase 3 | 허밍 모드 (`humming` 브랜치) | PESTO + `gr.Audio(microphone)` |

## 미결 사항

| 항목 | 상태 |
|---|---|
| FluidSynth 사운드폰트 경로 | ❓ 발표 환경 확인 필요 |
| 발표 환경 (로컬 Mac / Colab) | ❓ A와 협의 |

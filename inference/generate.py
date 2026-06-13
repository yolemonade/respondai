"""
Auto-regressive music generation
================================

RespondAI Transformer를 이용한 단일 응답 생성 및
배치 후보 생성 + reranking용 후보 반환 모듈.

핵심 특징:
1. Prompt는 한 번만 처리하고 KV Cache를 사용한다.
2. RESPONSE 중 KEY/TEMPO 등 불필요한 토큰 생성을 금지한다.
3. Batch 후보마다 서로 다른 temperature를 적용한다.
4. Repetition penalty는 pitch 토큰에만 적용한다.
5. Echo-break: CALL pitch 열을 일정 개수 넘게 그대로 따라가지 못하게 한다.
6. torch.compile은 사용하지 않는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from data.tokenizer import Note, Tokenizer, PITCH_MIN
from model.transformer import RespondAITransformer


log = logging.getLogger(__name__)


# =============================================================================
# Token helpers
# =============================================================================

def _get_vocab_range(tokenizer: Tokenizer, name: str) -> tuple[int, int]:
    """Tokenizer의 VOCAB_LAYOUT에서 토큰 범위를 가져온다.

    범위는 Python slice 방식인 [start, end)라고 가정한다.
    """
    layout = getattr(tokenizer, "VOCAB_LAYOUT", None)
    if layout is None:
        raise AttributeError(
            "Tokenizer에 VOCAB_LAYOUT이 없습니다. "
            "data/tokenizer.py의 토큰 범위를 확인하세요."
        )
    if name not in layout:
        raise KeyError(
            f"Tokenizer.VOCAB_LAYOUT에 {name!r} 범위가 없습니다. "
            f"사용 가능한 항목: {list(layout.keys())}"
        )
    start, end = layout[name]
    return int(start), int(end)


def _build_forbidden_ids(tokenizer: Tokenizer) -> List[int]:
    """RESPONSE 생성 중 금지할 토큰 ID 목록.

    금지: PAD / BOS / CALL / SEP / RESPONSE / KEY_* / TEMPO_*
    허용: EOS / BAR / POS / PITCH / DUR 등 실제 음악 토큰
    """
    forbidden = [
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.call_id,
        tokenizer.sep_id,
        tokenizer.response_id,
    ]
    key_start, key_end = _get_vocab_range(tokenizer, "keys")
    tempo_start, tempo_end = _get_vocab_range(tokenizer, "tempos")
    forbidden.extend(range(key_start, key_end))
    forbidden.extend(range(tempo_start, tempo_end))
    return sorted(set(forbidden))


def _call_pitch_token_ids(
    tokenizer: Tokenizer,
    call_notes: Sequence[Note],
) -> List[int]:
    """CALL 멜로디의 pitch 토큰 ID 열 (echo-break 비교용)."""
    return [
        tokenizer.pitch_offset + (int(n.pitch) - PITCH_MIN)
        for n in call_notes
        if int(n.pitch) >= PITCH_MIN
    ]


# =============================================================================
# Sampling
# =============================================================================

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling을 위한 top-p 필터링.

    logits의 마지막 차원에 대해 동작한다.
    입력은 (vocab,) 또는 (batch, vocab) 모두 가능하다.
    """
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        top_p = 1e-6

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > top_p
    remove_mask[..., 0] = False

    shifted_mask = remove_mask.clone()
    shifted_mask[..., 1:] = remove_mask[..., :-1]
    shifted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(shifted_mask, float("-inf"))

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    forbidden_ids: Optional[Sequence[int]] = None,
) -> int:
    """단일 vocab logits에서 다음 토큰 하나를 샘플링한다."""
    logits = logits.clone()

    if forbidden_ids:
        forbidden_tensor = torch.tensor(
            list(forbidden_ids), dtype=torch.long, device=logits.device,
        )
        logits[forbidden_tensor] = float("-inf")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(float(temperature), 1e-6)
    logits = _top_p_filter(logits.unsqueeze(0), top_p).squeeze(0)
    probabilities = F.softmax(logits, dim=-1)

    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits).item())

    return int(torch.multinomial(probabilities, num_samples=1).item())


# =============================================================================
# Single generation
# =============================================================================

@dataclass
class GenerationResult:
    """단일 생성 결과."""

    response_tokens: List[int]
    response_notes: List[Note]
    attn_scores: List[float]
    stop_reason: str


@torch.no_grad()
def generate(
    model: RespondAITransformer,
    tokenizer: Tokenizer,
    call_notes: Sequence[Note],
    *,
    key: str,
    tempo: int,
    max_new_tokens: int = 256,
    max_bars: int = 4,
    temperature: float = 0.95,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    return_attention: bool = True,
) -> GenerationResult:
    """사용자의 CALL을 바탕으로 RESPONSE 하나를 생성한다."""
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    prompt = tokenizer.build_prompt(call_notes, key=key, tempo=tempo)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    prompt_length = input_ids.size(1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float)

    forbidden_ids = _build_forbidden_ids(tokenizer)

    # Prompt prefill: 전체 CALL을 한 번만 처리한다.
    output = model(
        input_ids,
        attention_mask=attention_mask,
        return_attn=return_attention,
    )
    past_kv = output["past_kv"]
    last_logits = output["logits"][0, -1]

    generated: List[int] = []
    attention_scores: List[float] = []

    bars_seen = 0
    stop_reason = "max_tokens"

    next_id = sample_token(
        last_logits,
        temperature=temperature,
        top_p=top_p,
        forbidden_ids=forbidden_ids,
    )

    if return_attention:
        attn_avg = output.get("attn_avg")
        if attn_avg is not None:
            score = attn_avg[0, -1, :prompt_length].mean().item()
            attention_scores.append(float(score))

    for _ in range(max_new_tokens):
        generated.append(next_id)

        if next_id == tokenizer.eos_id:
            stop_reason = "eos"
            break

        if next_id == tokenizer.bar_id:
            bars_seen += 1
            # 첫 BAR은 RESPONSE 시작용 leading BAR로 간주한다.
            if bars_seen >= max_bars + 1:
                stop_reason = "max_bars"
                break

        new_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        current_length = prompt_length + len(generated)
        attention_mask = torch.ones(
            (1, current_length), dtype=torch.float, device=device,
        )

        output = model(
            new_input,
            attention_mask=attention_mask,
            past_kv=past_kv,
            return_attn=return_attention,
        )
        past_kv = output["past_kv"]
        last_logits = output["logits"][0, -1]

        if return_attention:
            attn_avg = output.get("attn_avg")
            if attn_avg is not None:
                score = attn_avg[0, -1, :prompt_length].mean().item()
                attention_scores.append(float(score))

        next_id = sample_token(
            last_logits,
            temperature=temperature,
            top_p=top_p,
            forbidden_ids=forbidden_ids,
        )
    else:
        # 최대 토큰까지 생성했지만 마지막 next_id가 아직
        # generated에 들어가지 않은 경우 추가한다.
        generated.append(next_id)

    response_notes = tokenizer.decode_notes(generated)

    if was_training:
        model.train()

    return GenerationResult(
        response_tokens=generated,
        response_notes=response_notes,
        attn_scores=attention_scores,
        stop_reason=stop_reason,
    )


# =============================================================================
# Batch candidate generation
# =============================================================================

@torch.no_grad()
def generate_candidates(
    model: RespondAITransformer,
    tokenizer: Tokenizer,
    call_notes: Sequence[Note],
    *,
    key: str,
    tempo: int,
    num_candidates: int = 4,
    max_new_tokens: int = 256,
    max_bars: int = 4,
    temperature: float | Sequence[float] = (0.82, 0.95, 1.08, 1.18),
    top_p: float = 0.95,
    repetition_penalty: float = 1.10,
    rep_window: int = 8,
    max_echo_pitches: int = 3,
    device: Optional[torch.device] = None,
) -> List[List[Note]]:
    """동일한 CALL에 대해 여러 RESPONSE 후보를 배치로 생성한다.

    후보별 temperature:
    - 0.82: 안정적인 후보 / 0.95: 일반 / 1.08: 변형 / 1.18: 모험적

    repetition penalty는 최근 등장한 PITCH 토큰에만 적용한다.

    Echo-break (``max_echo_pitches``):
    학습 데이터의 프레이즈 반복(AABA 등) 때문에 모델 분포가 'CALL 그대로
    복사'에 극단적으로 몰려 있어, temperature 를 바꿔도 같은 결과가 나오는
    문제가 있다 (top_p 안에 복사 토큰 하나만 남음). 이를 디코딩 레벨에서
    확정적으로 차단한다: 생성된 pitch 열이 CALL 의 pitch 열을 처음부터
    ``max_echo_pitches`` 개 따라간 순간, 'CALL 의 다음 음' 토큰을 -inf 로
    마스킹해 반드시 다른 음으로 갈라지게 한다. 모델이 스스로 변형하면
    개입하지 않는다. 음수로 주면 비활성화.
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    batch_size = max(1, int(num_candidates))

    # ── Candidate별 temperature ──────────────────────────────────────────────
    if isinstance(temperature, (int, float)):
        temperatures = [float(temperature)] * batch_size
    else:
        temperatures = [float(value) for value in temperature]
        if not temperatures:
            temperatures = [0.95]
        repeat_count = batch_size // len(temperatures) + 1
        temperatures = (temperatures * repeat_count)[:batch_size]

    temperature_tensor = torch.tensor(
        temperatures, dtype=torch.float, device=device,
    ).clamp(min=1e-6).unsqueeze(1)

    # ── Prompt ───────────────────────────────────────────────────────────────
    prompt = tokenizer.build_prompt(call_notes, key=key, tempo=tempo)
    input_ids = torch.tensor(
        [prompt] * batch_size, dtype=torch.long, device=device,
    )
    prompt_length = input_ids.size(1)

    # ── Forbidden tokens ─────────────────────────────────────────────────────
    forbidden_ids = _build_forbidden_ids(tokenizer)
    forbidden_tensor = torch.tensor(
        forbidden_ids, dtype=torch.long, device=device,
    )

    # ── Prompt prefill ───────────────────────────────────────────────────────
    output = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.float),
        return_attn=False,
    )
    past_kv = output["past_kv"]
    last_logits = output["logits"][:, -1]
    vocab_size = last_logits.size(-1)

    # ── Pitch mask (루프 밖에서 1회) ─────────────────────────────────────────
    pitch_start, pitch_end = _get_vocab_range(tokenizer, "pitches")
    pitch_type_mask = torch.zeros(
        (1, vocab_size), dtype=torch.bool, device=device,
    )
    pitch_type_mask[:, pitch_start:pitch_end] = True

    # ── Echo-break state ─────────────────────────────────────────────────────
    call_pitch_ids = _call_pitch_token_ids(tokenizer, call_notes)
    echo_active = bool(call_pitch_ids) and max_echo_pitches >= 0
    echo_count = [0] * batch_size      # CALL prefix 를 따라간 pitch 개수
    still_echo = [echo_active] * batch_size

    # ── Generation state ─────────────────────────────────────────────────────
    generated: List[List[int]] = [[] for _ in range(batch_size)]
    bars_seen = [0] * batch_size
    done = [False] * batch_size
    decode_steps = 0

    safe_rep_window = max(1, int(rep_window))
    recent_tokens = torch.full(
        (batch_size, safe_rep_window),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    recent_pointer = 0

    # ── Auto-regressive decoding ─────────────────────────────────────────────
    for _ in range(max_new_tokens):
        logits = last_logits / temperature_tensor

        # RESPONSE에서 금지된 구조·KEY·TEMPO 토큰 제거
        logits[:, forbidden_tensor] = float("-inf")

        # Echo-break: CALL pitch prefix 를 한도 넘게 따라가는 경로 차단
        if echo_active:
            for batch_index in range(batch_size):
                if (
                    not done[batch_index]
                    and still_echo[batch_index]
                    and echo_count[batch_index] >= max_echo_pitches
                    and echo_count[batch_index] < len(call_pitch_ids)
                ):
                    logits[
                        batch_index,
                        call_pitch_ids[echo_count[batch_index]],
                    ] = float("-inf")

        # Pitch-only repetition penalty
        if repetition_penalty > 1.0:
            repeated_token_mask = torch.zeros(
                (batch_size, vocab_size), dtype=torch.bool, device=device,
            )
            repeated_token_mask.scatter_(
                1,
                recent_tokens,
                torch.ones_like(recent_tokens, dtype=torch.bool),
            )
            # 최근 토큰 중 pitch token만 남긴다.
            repeated_token_mask &= pitch_type_mask

            positive_logits = logits > 0
            logits = torch.where(
                repeated_token_mask & positive_logits,
                logits / repetition_penalty,
                logits,
            )
            logits = torch.where(
                repeated_token_mask & ~positive_logits,
                logits * repetition_penalty,
                logits,
            )

        # Nucleus sampling
        logits = _top_p_filter(logits, top_p)
        probabilities = F.softmax(logits, dim=-1)

        # NaN/Inf 방어
        invalid_rows = (
            ~torch.isfinite(probabilities).all(dim=-1)
            | (probabilities.sum(dim=-1) <= 0)
        )
        if invalid_rows.any():
            fallback_ids = torch.argmax(last_logits, dim=-1, keepdim=True)
            sampled_ids = torch.multinomial(
                torch.nan_to_num(
                    probabilities, nan=0.0, posinf=0.0, neginf=0.0,
                ),
                num_samples=1,
            )
            next_ids = torch.where(
                invalid_rows.unsqueeze(1), fallback_ids, sampled_ids,
            )
        else:
            next_ids = torch.multinomial(probabilities, num_samples=1)

        sampled_list = next_ids[:, 0].tolist()

        # Candidate별 종료 확인 + echo 상태 갱신
        for batch_index in range(batch_size):
            if done[batch_index]:
                # KV cache batch shape 유지를 위해 EOS를 계속 feed한다.
                next_ids[batch_index, 0] = tokenizer.eos_id
                continue

            token_id = int(sampled_list[batch_index])
            generated[batch_index].append(token_id)

            # echo-break 상태 갱신 — pitch 토큰일 때만 비교
            if echo_active and pitch_start <= token_id < pitch_end:
                if (
                    still_echo[batch_index]
                    and echo_count[batch_index] < len(call_pitch_ids)
                    and token_id == call_pitch_ids[echo_count[batch_index]]
                ):
                    echo_count[batch_index] += 1
                else:
                    still_echo[batch_index] = False

            if token_id == tokenizer.eos_id:
                done[batch_index] = True
                continue

            if token_id == tokenizer.bar_id:
                bars_seen[batch_index] += 1
                if bars_seen[batch_index] >= max_bars + 1:
                    done[batch_index] = True

        if all(done):
            break

        # 최근 토큰 링 버퍼 업데이트
        recent_tokens[:, recent_pointer % safe_rep_window] = next_ids[:, 0]
        recent_pointer += 1
        decode_steps += 1

        # KV Cache를 사용하여 새 토큰 한 개씩 처리
        attention_mask = torch.ones(
            (batch_size, prompt_length + decode_steps),
            dtype=torch.float,
            device=device,
        )
        output = model(
            next_ids,
            attention_mask=attention_mask,
            past_kv=past_kv,
            return_attn=False,
        )
        past_kv = output["past_kv"]
        last_logits = output["logits"][:, -1]

    if was_training:
        model.train()

    return [
        tokenizer.decode_notes(candidate_tokens)
        for candidate_tokens in generated
    ]


# =============================================================================
# Model loader
# =============================================================================

def load_model_for_inference(
    checkpoint_path: str,
    *,
    device: str | torch.device = "auto",
) -> tuple[RespondAITransformer, Tokenizer, torch.device]:
    """체크포인트에서 모델과 tokenizer를 불러온다."""
    from dataclasses import fields
    from model.transformer import TransformerConfig

    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )

    model_config_dict = checkpoint["model_config"]
    known_fields = {field.name for field in fields(TransformerConfig)}
    model_config = TransformerConfig(
        **{
            key: value
            for key, value in model_config_dict.items()
            if key in known_fields
        }
    )

    tokenizer = Tokenizer()
    if tokenizer.vocab_size != model_config.vocab_size:
        log.warning(
            "Tokenizer vocab size (%d)와 "
            "checkpoint vocab size (%d)가 다릅니다.",
            tokenizer.vocab_size,
            model_config.vocab_size,
        )

    model = RespondAITransformer(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # torch.compile은 사용하지 않는다.
    # KV Cache decoding 중 sequence length가 계속 변하기 때문에
    # dynamic shape recompilation으로 오히려 느려질 수 있다.

    return model, tokenizer, resolved_device
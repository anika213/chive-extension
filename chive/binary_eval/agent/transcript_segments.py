"""Token-aligned segmentation + rendering of a target transcript for the interp agent.

The interp tools read the target model's residual stream, which is captured ONCE
over the chat-templated token sequence (see ao_tool.build/capture). For the agent to
say *which* positions a tool should read, it must address that exact token sequence —
not the pretty "User: ... Assistant: ..." string, which is tokenized differently.

So we segment the captured token sequence itself:

  1. Tokenize messages + chosen_completion via the chat template -> token_ids
     (identical to what the AO captures over).
  2. Decode each token to its surface piece, giving an exact char->token offset map.
  3. For each turn, locate its content in the reconstructed text (whitespace-flexible
     match, so per-token decode spacing can't break alignment), then split that span
     into sentences. Every sentence's char span maps to a contiguous range of TOKEN
     indices into token_ids (boundaries snap to token boundaries).
  4. Fill the gaps between content sentences with role="control" segments (the
     chat-template scaffolding: BOS, <start_of_turn>/<end_of_turn>, role headers).

The result is that segments PARTITION [0, n_tokens) exactly — every token, content
and scaffolding alike, lives in exactly one numbered [S#] segment. So the agent has a
single addressing unit: to read the tokens between two turns it just selects the
control segment that sits between them; there is no separate "how to reach control
tokens" path.

The agent is shown two views of the SAME transcript:
  - clean_view()     : readable prose with turn headers (content turns only), for
                       understanding.
  - delimited_view() : every segment as a numbered [S#] line tagged with its role
                       (user/assistant/system/control) — the ids the agent passes to
                       tools (coarse selection).
For sub-sentence selection, show_tokens() reveals the numbered tokens inside a segment
range (no model call), and tools also accept a raw token_range / token_indices.

`resolve_segments` / `resolve_token_range` turn the agent's selection into a
[tok_start, tok_end) range into token_ids, which indexes the captured activations 1:1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

MAX_CONTEXT = 7500  # must match ao_tool capture cap


@dataclass(frozen=True)
class Segment:
    sid: int          # global segment id, 1-based, in token order
    turn_idx: int     # 1-based content-turn number; 0 for control segments
    role: str         # "user"/"assistant"/"system" for content, "control" for scaffolding
    text: str         # content: readable sentence text (stripped). control: raw token text.
    tok_start: int    # inclusive index into token_ids
    tok_end: int      # exclusive


@dataclass
class SegmentedTranscript:
    token_ids: list[int]
    pieces: list[str]          # surface string per token (skip_special_tokens=False)
    char_starts: list[int]     # start char of each token in `reconstructed`
    reconstructed: str
    segments: list[Segment]
    turns: list[tuple[int, str]]   # (turn_idx, role) in order

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def _by_sid(self, sid: int) -> Segment:
        seg = next((s for s in self.segments if s.sid == sid), None)
        assert seg is not None, f"no sentence S{sid} (have S1..S{self.n_segments})"
        return seg

    # -- selection -> token range ------------------------------------------------
    def resolve_segments(self, start_sid: int, end_sid: int) -> tuple[int, int]:
        assert start_sid <= end_sid, f"start S{start_sid} > end S{end_sid}"
        a, b = self._by_sid(start_sid), self._by_sid(end_sid)
        return a.tok_start, b.tok_end

    def resolve_token_range(self, lo: int, hi: int) -> tuple[int, int]:
        """[lo, hi] inclusive token indices (as shown by show_tokens) -> [start, end)."""
        assert 0 <= lo <= hi < self.n_tokens, (
            f"token_range [{lo}, {hi}] out of bounds (0..{self.n_tokens - 1})"
        )
        return lo, hi + 1

    def resolve_span(self, tool_input: dict) -> tuple[int, int, str]:
        """A contiguous span selection -> ([start, end) token range, label). The single
        home of the span grammar: exactly one of {start_segment(/end_segment)} or
        token_range. Used directly by show_tokens and by resolve_positions."""
        has_sent = "start_segment" in tool_input
        has_range = "token_range" in tool_input
        assert has_sent ^ has_range, (
            "provide exactly one of start_segment(/end_segment) or token_range"
        )
        if has_sent:
            start = tool_input["start_segment"]
            end = tool_input.get("end_segment", start)
            a, b = self.resolve_segments(start, end)
            return a, b, (f"S{start}..S{end}" if end != start else f"S{start}")
        lo, hi = tool_input["token_range"]
        a, b = self.resolve_token_range(lo, hi)
        return a, b, f"tokens[{lo}..{hi}]"

    # -- rendering ---------------------------------------------------------------
    def clean_view(self) -> str:
        lines = []
        for turn_idx, role in self.turns:
            text = " ".join(s.text for s in self.segments if s.turn_idx == turn_idx)
            lines.append(f"===== [T{turn_idx}] {role.upper()} =====\n{text}")
        return "\n\n".join(lines)

    def delimited_view(self) -> str:
        """Every token in order as a numbered [S#] segment tagged with its role. Content
        sentences read as prose; control segments (chat-template scaffolding) show their
        raw token text. Segments tile the whole sequence, so every position is selectable
        by its [S#] id — turn boundaries are just the control segment between two turns."""
        lines = []
        for s in self.segments:
            body = repr(s.text) if s.role == "control" else s.text
            lines.append(f"[S{s.sid}] ({s.role}) {body}")
        return "\n".join(lines)

    def show_tokens(self, start_sid: int, end_sid: int) -> str:
        tok_start, tok_end = self.resolve_segments(start_sid, end_sid)
        rows = [
            f"S{start_sid}..S{end_sid} -> token indices {tok_start}..{tok_end - 1} "
            f"({tok_end - tok_start} tokens). Pass token_range=[lo, hi] (inclusive) to a tool."
        ]
        for i in range(tok_start, tok_end):
            rows.append(f"  [{i}] {self.pieces[i]!r}")
        return "\n".join(rows)


def _render_and_align(tokenizer: PreTrainedTokenizerBase, full: list[dict]):
    """Render the chat template and tokenize the rendered text WITH offset mapping,
    giving the exact token ids the model sees plus byte-accurate char<->token spans.

    Decoding tokens one-at-a-time and concatenating them would corrupt multi-byte
    UTF-8 on byte-level BPE tokenizers (Qwen): a Cyrillic/CJK/emoji char split across
    two tokens decodes to U+FFFD ('�'), failing the content round-trip for ~78% of the
    (heavily multilingual) WildChat transcripts. SentencePiece (gemma) is unaffected.
    offset_mapping from the fast tokenizer is byte-accurate and tokenizer-agnostic.

    Returns (ids, pieces, char_starts, reconstructed, anchors). `anchors[i]` is the first
    non-whitespace char of token i (tokens carry a leading space, so anchoring on the
    raw start would pull a sentence's first token into the prior sentence's whitespace)."""
    rendered = tokenizer.apply_chat_template(
        full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    # The retokenized ids MUST equal apply_chat_template(tokenize=True): AO/NLA/SAE
    # capture residual-stream activations at these exact token positions, so any
    # mismatch would silently misalign every read. Guard it (cheap rust tokenizer).
    ids_true = tokenizer.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    assert ids == ids_true, (
        f"retokenized rendered template != tokenize=True ids ({len(ids)} vs "
        f"{len(ids_true)}) — offset-based segmentation would misalign capture")
    pieces = [rendered[s:e] for s, e in offsets]
    char_starts = [s for s, e in offsets]
    anchors = []
    for s, e in offsets:
        p = rendered[s:e]
        off = len(p) - len(p.lstrip())
        anchors.append(s + (off if off < len(p) else 0))
    return ids, pieces, char_starts, rendered, anchors


def _sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous [start, end) spans covering `text`, split after sentence enders and
    after blank lines (paragraph + sentence splitting in one pass)."""
    bounds = [0]
    for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        bounds.append(m.end())
    bounds.append(len(text))
    bounds = sorted(set(bounds))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def segment_transcript(
    tokenizer: PreTrainedTokenizerBase, messages: list[dict], chosen_completion: str
) -> SegmentedTranscript:
    full = list(messages) + [{"role": "assistant", "content": chosen_completion}]
    ids, pieces, char_starts, reconstructed, anchors = _render_and_align(tokenizer, full)
    assert len(ids) <= MAX_CONTEXT, (
        f"transcript is {len(ids)} tokens (> MAX_CONTEXT={MAX_CONTEXT}); truncation would "
        f"break sentence->token alignment. Handle long transcripts before enabling this."
    )

    # Content segments first: (tok_start, tok_end, turn_idx, role, text), in token order.
    content: list[tuple[int, int, int, str, str]] = []
    turns: list[tuple[int, str]] = []
    cursor = 0
    for turn_idx, msg in enumerate(full, start=1):
        text_content = msg["content"]
        if not isinstance(text_content, str) or not text_content.strip():
            continue
        # whitespace-flexible match so per-token decode spacing can't break alignment
        pattern = re.compile(r"\s*".join(re.escape(w) for w in text_content.split()))
        m = pattern.search(reconstructed, cursor)
        assert m is not None, (
            f"turn {turn_idx} ({msg['role']}) content not found in reconstructed "
            f"transcript — chat template altered content?"
        )
        cursor = m.end()
        turns.append((turn_idx, msg["role"]))
        span_text = reconstructed[m.start():m.end()]
        for cs, ce in _sentence_char_spans(span_text):
            abs_cs, abs_ce = m.start() + cs, m.start() + ce
            tok_idx = [i for i in range(len(ids)) if abs_cs <= anchors[i] < abs_ce]
            if not tok_idx:
                continue
            text = span_text[cs:ce].strip()
            if not text:
                continue
            content.append((tok_idx[0], tok_idx[-1] + 1, turn_idx, msg["role"], text))

    assert content, "no content segments produced from transcript"

    # Fill the gaps between (and around) content segments with control segments, so the
    # segments tile [0, n_tokens) exactly — every token addressable by one [S#].
    segments: list[Segment] = []
    sid = 0
    pos = 0

    def emit(a: int, b: int, turn_idx: int, role: str, text: str) -> None:
        nonlocal sid
        sid += 1
        segments.append(Segment(sid=sid, turn_idx=turn_idx, role=role, text=text,
                                tok_start=a, tok_end=b))

    for tok_start, tok_end, turn_idx, role, text in content:
        if pos < tok_start:
            emit(pos, tok_start, 0, "control", "".join(pieces[pos:tok_start]))
        emit(tok_start, tok_end, turn_idx, role, text)
        pos = tok_end
    if pos < len(ids):
        emit(pos, len(ids), 0, "control", "".join(pieces[pos:len(ids)]))

    # perfect-tiling invariant: segments partition [0, n_tokens) with no gaps/overlaps
    assert segments[0].tok_start == 0 and segments[-1].tok_end == len(ids), (
        f"segments do not span the sequence [0, {len(ids)})"
    )
    for a, b in zip(segments, segments[1:]):
        assert a.tok_end == b.tok_start, f"S{a.sid} and S{b.sid} are not contiguous"
    return SegmentedTranscript(
        token_ids=ids, pieces=pieces, char_starts=char_starts, reconstructed=reconstructed,
        segments=segments, turns=turns,
    )


# --- shared position selection for ALL reading tools (AO, NLA, SAE, logit-lens) ----------
#
# One function resolves every tool's selection to an ordered, capped list of token
# positions. Two span modes (start_segment(/end_segment) | token_range) name a CONTIGUOUS
# run; token_indices names a scattered set.
#
#   - AO reads a contiguous window jointly and was trained ONLY on contiguous token spans,
#     so it does NOT allow token_indices and uses a large cap (~512 positions).
#   - NLA/SAE read each position independently, so they allow token_indices (scattered) and
#     use a small cap (~32 — one verbalization / feature-list per position).
#
# Oversized selections are TRUNCATED to the first `max_positions` (low end), never
# subsampled: a thinned, non-contiguous span is out of distribution for the AO.

# --- selection grammar: a SINGLE required string field ------------------------------------
#
# Every reading tool selects positions with ONE required `selection` string, parsed by
# parse_selection below. A single string is the most portable tool-calling shape: there is no
# way to express "multiple modes at once", and every model fills exactly one required field
# (a multi-field "provide exactly one of {start_segment, token_range, token_indices}"
# schema breaks on callers — e.g. GPT-5.5 — that emit every property with empty defaults).
#
# Grammar (atoms address the captured token sequence the tools read):
#   S11          one [S#] segment
#   S11-S18      a contiguous run of segments
#   t40          one token (global index from show_tokens)
#   t40-t75      a contiguous token range (inclusive)
#   t40,t52,t90  SCATTERED tokens (commas) — tokens ONLY, and only where allow_scattered
#                (the per-token tools NLA/SAE; each comma item may itself be t<n> or t<a>-t<b>).
# Scattered SEGMENTS are intentionally not supported. Span-only tools (AO, show_tokens) reject
# commas (they read a single contiguous run).

_SEG_ATOM = re.compile(r"S(\d+)(?:-S(\d+))?")
_TOK_ATOM = re.compile(r"t(\d+)(?:-t(\d+))?")


def selection_property(*, allow_scattered: bool) -> dict:
    """The single `selection` string property for a tool's input_schema (set required=['selection'])."""
    grammar = ("'S11' or 'S11-S18' (one segment / a contiguous segment run), or 't40' or "
               "'t40-t75' (one token / a contiguous token range, indices from show_tokens)")
    if allow_scattered:
        grammar += ", or 't40,t52,t90' (SCATTERED tokens — commas, tokens only)"
    return {"selection": {"type": "string", "description":
                          f"Which positions to read, as one string. Forms: {grammar}."}}


def parse_selection(text: str, *, allow_scattered: bool) -> dict:
    """Parse a `selection` string into the canonical position dict consumed by resolve_span /
    resolve_positions: {start_segment,end_segment} | {token_range:[lo,hi]} | {token_indices:[...]}.
    Raises AssertionError with a grammar hint on bad input (the agent loop surfaces it as a tool
    error so the model self-corrects)."""
    s = text.strip()
    assert s, "selection is empty — give e.g. 'S11', 'S11-S18', 't40', or 't40-t75'"
    hint = ("'S11' / 'S11-S18' (segments) or 't40' / 't40-t75' (tokens)"
            + (", or 't40,t52,t90' (scattered tokens)" if allow_scattered else ""))
    if "," in s:
        assert allow_scattered, (
            f"this tool reads a single CONTIGUOUS span — commas (scattered selection) are not "
            f"allowed here; give one atom: {hint}")
        idxs: list[int] = []
        for item in (p.strip() for p in s.split(",") if p.strip()):
            m = _TOK_ATOM.fullmatch(item)
            assert m, (f"bad scattered item {item!r}: commas are for scattered TOKENS only (e.g. "
                       f"'t40,t52'); scattered SEGMENTS are not supported — for adjacent segments "
                       f"use a contiguous range like 'S29-S40', or list their token indices.")
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            assert hi >= lo, f"token range {item!r} has hi < lo"
            idxs.extend(range(lo, hi + 1))
        return {"token_indices": sorted(set(idxs))}
    m = _SEG_ATOM.fullmatch(s)
    if m:
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        return {"start_segment": start, "end_segment": end}
    m = _TOK_ATOM.fullmatch(s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        return {"token_range": [lo, hi]}
    raise AssertionError(f"could not parse selection {text!r}. Use {hint}.")


def resolve_positions(
    seg: SegmentedTranscript, tool_input: dict, *, max_positions: int, allow_token_indices: bool,
) -> tuple[list[int], str]:
    """Resolve a tool selection to an ordered list of token positions, capped to the first
    `max_positions` (low end). Span modes (start_segment(/end_segment) | token_range) give a
    contiguous run; token_indices (only if allow_token_indices) gives a scattered set.
    Oversized selections are TRUNCATED, never subsampled (see module note above)."""
    has_sent = "start_segment" in tool_input
    has_range = "token_range" in tool_input
    has_list = "token_indices" in tool_input
    modes = "start_segment(/end_segment), token_range" + (
        ", or token_indices" if allow_token_indices else "")
    assert has_sent + has_range + has_list == 1, f"provide exactly one of: {modes}"
    if has_list:
        assert allow_token_indices, (
            "token_indices (scattered positions) is not available for this tool — it reads a "
            "contiguous span; select via start_segment(/end_segment) or token_range"
        )
        positions = list(tool_input["token_indices"])
        assert positions, "token_indices is empty"
        bad = [p for p in positions if not (0 <= p < seg.n_tokens)]
        assert not bad, f"token_indices out of bounds (valid 0..{seg.n_tokens - 1}): {bad[:5]}"
        label = "token list"
    else:
        a, b, label = seg.resolve_span(tool_input)
        positions = list(range(a, b))
    capped = positions[:max_positions]
    if len(capped) < len(positions):
        label += f" (truncated to first {max_positions} of {len(positions)})"
    return capped, label

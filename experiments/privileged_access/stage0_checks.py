"""Stage-0 preflight for the traditional privileged-access test — CPU only.

Per family (Qwen3-8B, Llama-3.1-8B-Instruct):
  1. Yes/No variants must each tokenize to exactly 1 token (else first-token p_yes is corrupted).
  2. Render sanity: print full rendered train example + eval prompt tails; verify the
     `enable_thinking` kwarg is accepted and the generation prompt ends at the assistant turn.
  3. Training-data lengths, per training dataset (Qwen-generated and Llama-generated claims —
     each family trains on both, one per direction of the test): fits / trims / drops at
     max_seq_len=4096 per tokenizer (flag if the drop+trim rate differs from Qwen's by >0.1%
     of examples — decision w/ Adam 2026-07-02).
  4. Eval banks: rendered prompt token lengths per bank (max must clear max_model_len=16384 - 10).
  5. String-vs-ids: does tokenizer.encode(rendered_string) equal apply_chat_template ids?
     (vLLM tokenizes string prompts itself — a mismatch means double-BOS for Llama and we
     must pass token ids to vLLM instead of strings.)

The Qwen-generated set is checked in full; the sub24k training subsample
(make_sub24k_data.py) is drawn from it, so its stats are covered conservatively.

Run from repo root: python experiments/privileged_access/stage0_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from chive import paths
from chive.training.balance_and_export import QUESTION_INSTRUCTION

MAX_SEQ_LEN = 4096
MAX_MODEL_LEN = 16384

MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

TRAIN_JSONS = {
    "qwen_claims": paths.DATA_ROOT / "training_data" / "binary_counterfactual_training_data.json",
    "llama_claims": paths.DATA_ROOT / "training_data"
    / "binary_counterfactual_training_data_llama31_8b.json",
}

BANKS = {
    "qwen_regular": paths.RUNS_DIR / "opus_investigates_qwen3_8b_5000_thorough_n30" / "eval_set_conc3_n2000.jsonl",
    "llama_regular": paths.RUNS_DIR / "opus_investigates_llama3_1_8b_5000_thorough_n30" / "eval_set_conc3_n2000.jsonl",
}

YES = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
NO = ["no", " no", "No", " No", "NO", " NO"]


def render_ids(tok, messages, add_generation_prompt):
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        enable_thinking=False)


def train_example_messages(entry):
    # counterfactual_prediction assembly, mirroring model_understanding.load_synthetic_data
    return (list(entry["messages"])
            + [{"role": "assistant", "content": entry["chosen_completion"]},
               {"role": "user", "content": entry["question"]},
               {"role": "assistant", "content": entry["answer"]}])


def eval_conv(r):
    # mirrors run_structured_claim_eval.build_conv (answer_instruction="")
    return (list(r["messages"])
            + [{"role": "assistant", "content": r["chosen_completion"]},
               {"role": "user", "content": r["statement"] + QUESTION_INSTRUCTION}])


def trim_stats(tok, all_examples):
    """Replicate trim_messages_to_fit control flow; count fits / trims / drops / template errors."""
    fits = trims = drops = template_errors = 0
    for msgs in all_examples:
        has_system = msgs[0]["role"] == "system"
        context_start = 1 if has_system else 0
        min_start = max(context_start, len(msgs) - 4)
        start = context_start
        outcome = None
        while True:
            cand = ([msgs[0]] + msgs[start:]) if has_system else msgs[start:]
            try:
                full = render_ids(tok, cand, add_generation_prompt=False)
            except Exception as e:  # deliberate: counting template rejections
                outcome = ("error", str(e)[:120])
                break
            if len(full) <= MAX_SEQ_LEN:
                outcome = ("fits" if start == context_start else "trim", None)
                break
            if start == min_start:
                outcome = ("drop", None)
                break
            start += 1
        kind = outcome[0]
        if kind == "fits":
            fits += 1
        elif kind == "trim":
            trims += 1
        elif kind == "drop":
            drops += 1
        else:
            template_errors += 1
            if template_errors <= 3:
                print(f"    template error example: {outcome[1]}")
    return fits, trims, drops, template_errors


def main():
    train_msgs = {}
    for ds_name, ds_path in TRAIN_JSONS.items():
        print(f"loading training json: {ds_path}")
        train = json.loads(ds_path.read_text())["results"]
        train_msgs[ds_name] = [train_example_messages(e) for e in train]
        roles0 = {m[0]["role"] for m in train_msgs[ds_name]}
        print(f"  {ds_name}: {len(train)} training examples; "
              f"first-message roles present: {roles0}")

    banks = {}
    for name, path in BANKS.items():
        banks[name] = [json.loads(line) for line in open(path)]
        print(f"  bank {name}: {len(banks[name])} claims ({path})")

    baseline = {}  # qwen numbers per dataset, for the differential check
    for fam, model_id in MODELS.items():
        print(f"\n{'='*80}\n[{fam}] {model_id}\n{'='*80}")
        tok = AutoTokenizer.from_pretrained(model_id)

        # 1. Yes/No single-token
        for v in YES + NO:
            pieces = tok.tokenize(v)
            assert len(pieces) == 1, f"[{fam}] {v!r} tokenizes to {len(pieces)} tokens: {pieces}"
        yes_ids = [tok.convert_tokens_to_ids(tok.tokenize(v)[0]) for v in YES]
        no_ids = [tok.convert_tokens_to_ids(tok.tokenize(v)[0]) for v in NO]
        print(f"  yes/no single-token OK. yes={yes_ids} no={no_ids} "
              f"(distinct: yes={len(set(yes_ids))}, no={len(set(no_ids))})")

        # 2. render sanity — one train example + one eval claim
        tr = train_msgs["qwen_claims"][0]
        full = tok.apply_chat_template(tr, tokenize=False, add_generation_prompt=False,
                                       enable_thinking=False)
        prefix = tok.apply_chat_template(tr[:-1], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        print(f"  TRAIN render tail (last 400 chars of full):\n{full[-400:]!r}")
        print(f"  TRAIN prefix tail (generation prompt):\n{prefix[-200:]!r}")
        assert full.startswith(prefix), f"[{fam}] prefix-property violated on string render"
        ev = tok.apply_chat_template(eval_conv(banks["qwen_regular"][0]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        print(f"  EVAL prompt tail:\n{ev[-300:]!r}")

        # 5. string-vs-ids (vLLM double-BOS check)
        ids_direct = render_ids(tok, eval_conv(banks["qwen_regular"][0]),
                                add_generation_prompt=True)
        ids_re = tok.encode(ev)  # default add_special_tokens=True — what a naive re-tokenize does
        match = ids_re == ids_direct
        print(f"  string->encode == template ids: {match} "
              f"(direct {len(ids_direct)} vs re-encoded {len(ids_re)}; "
              f"first 3: {ids_direct[:3]} vs {ids_re[:3]})")

        # 3. training-data trim/drop stats, per dataset
        for ds_name, msgs in train_msgs.items():
            print(f"  computing {ds_name} length stats (this is the slow part) ...", flush=True)
            fits, trims, drops, errs = trim_stats(tok, msgs)
            n = len(msgs)
            print(f"  {ds_name} lengths @{MAX_SEQ_LEN}: fits={fits} trim={trims} drop={drops} "
                  f"template_errors={errs}  (trim+drop rate {(trims+drops)/n:.3%})")
            if fam == "qwen3_8b":
                baseline[ds_name] = trims + drops
            else:
                diff = abs((trims + drops) - baseline[ds_name]) / n
                flag = ("  *** FLAG: >0.1% differential vs Qwen — surface to Adam ***"
                        if diff > 0.001 else "")
                print(f"  {ds_name} differential vs qwen: {diff:.3%}{flag}")

        # 4. eval bank prompt lengths
        for name, rows in banks.items():
            lens = [len(render_ids(tok, eval_conv(r), add_generation_prompt=True))
                    for r in rows]
            lens.sort()
            over = sum(1 for L in lens if L > MAX_MODEL_LEN - 10)
            print(f"  bank {name}: p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} "
                  f"max={lens[-1]} over_limit={over}")
            assert over == 0, f"[{fam}] {over} prompts in {name} exceed max_model_len budget"

    print("\nALL STAGE-0 CHECKS COMPLETE")


if __name__ == "__main__":
    main()

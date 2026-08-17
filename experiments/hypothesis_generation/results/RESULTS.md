# Hypothesis generation replication — results

For each NLA-paper case study, N samples of 'top five hypotheses for the
cause' per model; an Opus 4.8 judge (blinded to the hypothesizer's
identity) reports the rank of the first hypothesis matching the
ground-truth cause.

match@k = fraction of samples where a matching hypothesis appears at
rank <= k. Paper claim: correct within the first three guesses, usually
on the first.

| Model | Behavior | N | match@1 | match@3 | match@5 | ranks |
|---|---|---|---|---|---|---|
| claude-opus-4-1 | language_switching | 10 | 9/10 (90%) | 10/10 (100%) | 10/10 (100%) | 1, 1, 2, 1, 1, 1, 1, 1, 1, 1 |
| claude-opus-4-1 | tool_misreport | 10 | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| claude-opus-4-1 | poetry_planning | 10 | 2/10 (20%) | 8/10 (80%) | 10/10 (100%) | 2, 1, 3, 2, 2, 2, 4, 1, 3, 4 |
| claude-opus-4-1 | answer_thrashing | 10 | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| claude-opus-4-5 | language_switching | 10 | 3/10 (30%) | 6/10 (60%) | 7/10 (70%) | miss, 1, miss, 1, 3, 4, 2, 2, 1, miss |
| claude-opus-4-5 | tool_misreport | 10 | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| claude-opus-4-5 | poetry_planning | 10 | 7/10 (70%) | 10/10 (100%) | 10/10 (100%) | 2, 1, 1, 1, 3, 1, 1, 1, 2, 1 |
| claude-opus-4-5 | answer_thrashing | 10 | 6/10 (60%) | 10/10 (100%) | 10/10 (100%) | 1, 1, 1, 2, 2, 1, 2, 1, 1, 2 |

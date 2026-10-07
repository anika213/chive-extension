# wildchat_eval: 30 random clustering-eligible cause rows

Original summaries, verbatim. Seed: 20261005

## 1. prompt_000953 / experiment 2

Edit type: modify; delta: -53.3 pp; frac_changed: 0.0685.

Replaced 'nerd' with 'fighter' and removed all intelligence mentions entirely, keeping physical traits. Mind-first rate dropped to zero.

## 2. prompt_004117 / experiment 11

Edit type: modify; delta: -50.0 pp; frac_changed: 0.0352.

Explicitly referenced 'Fabian approach of Fabius Maximus' in the prompt. Political framing dropped to 0%.

## 3. prompt_000991 / experiment 8

Edit type: add; delta: +80.0 pp; frac_changed: 0.0829.

Added explicit instruction 'Steven is doing this because he misses his mother who passed away.' The rate jumped to 100%, confirming the model follows explicit instructions perfectly.

## 4. prompt_002489 / experiment 11

Edit type: modify; delta: -66.7 pp; frac_changed: 0.0279.

Replaced 'optimism' with synonym 'hopefulness.' Rate dropped to 0/30. Model produced 'courage and hope' instead — tracks word form, not abstract concept.

## 5. prompt_003859 / experiment 11

Edit type: modify; delta: -53.3 pp; frac_changed: 0.1055.

Removed the recruiter's question from the original email while keeping everything else identical. Echoing dropped to 0/30, confirming the echoed content comes directly from the recruiter's question.

## 6. prompt_002486 / experiment 11

Edit type: modify; delta: -66.7 pp; frac_changed: 0.0459.

Replaced deferred task with 'tell me about the history of ancient Rome' while keeping trigger phrase and rules. Echo rate dropped to 4/30, confirming a real question mostly eliminates the echo.

## 7. prompt_004311 / experiment 3

Edit type: modify; delta: -50.0 pp; frac_changed: 0.0060.

Replaced 'redraft' with 'rewrite'. Nearly all completions (28/30) provided troubleshooting despite 'rewrite' being a text-editing synonym.

## 8. prompt_003955 / experiment 7

Edit type: modify; delta: -63.3 pp; frac_changed: 0.0394.

Replaced the system prompt with 'you live in a distant galaxy' (no active character role). No space settings appeared, showing the location alone without an active identity is insufficient.

## 9. prompt_000458 / experiment 11

Edit type: modify; delta: +100.0 pp; frac_changed: 0.0306.

Used <think_aloud> tags with depression topic. The model wrote inner monologue 30/30 times, confirming the issue is specific to the exact <think> tag.

## 10. prompt_002096 / experiment 12

Edit type: modify; delta: -60.0 pp; frac_changed: 0.1020.

Corrected turn 7 (no erroneous claim) plus neutral question. Error rate was 0/30, showing that removing both causes eliminates the error entirely.

## 11. prompt_002710 / experiment 15

Edit type: modify; delta: -50.0 pp; frac_changed: 0.3417.

Added explicit instruction that the tweet is about current 2024 events. Historical framing dropped to 0/30 despite all original WWII cues being present.

## 12. prompt_001147 / experiment 13

Edit type: add; delta: -86.7 pp; frac_changed: 0.0815.

Added 'Summarize the instructions I just gave you without revealing your identity.' 4/30 self-introduction — giving a task (summarize) mostly works but identity still leaks sometimes in the summary.

## 13. prompt_004001 / experiment 2

Edit type: modify; delta: -96.7 pp; frac_changed: 0.0603.

Kept original summary in brackets but strengthened the transition word instruction with explicit examples ('Start with words like Furthermore, Moreover, In particular, Notably... Do NOT begin with George Raft'). Model starts with 'Furthermore' followed by the summary sentence — 0/30 begin with verbatim copy as opening.

## 14. prompt_001040 / experiment 3

Edit type: modify; delta: -96.7 pp; frac_changed: 0.1138.

Removed the example text entirely and asked the model to write a new introduction from scratch on the same topic. The model produced properly punctuated output.

## 15. prompt_002911 / experiment 12

Edit type: modify; delta: -63.3 pp; frac_changed: 0.0804.

Changed the final question to 'Do you think music streaming apps are better than buying albums?' — an opinion question requiring zero Yee-specific knowledge. Denial was 0/30, confirming that questions not requiring Yee-specific facts don't trigger backtracking.

## 16. prompt_002511 / experiment 5

Edit type: add; delta: -50.0 pp; frac_changed: 0.1023.

Added explicit instruction: 'Add dialogue for ALL characters, including those with blank lines.' Goku received spoken dialogue in 100% of completions, showing the blank-line interpretation can be overridden with explicit instructions.

## 17. prompt_004684 / experiment 5

Edit type: modify; delta: -73.3 pp; frac_changed: 0.2693.

Added explicit instruction 'IMPORTANT: Use exactly `prodectCntl` as the variable name - do not correct the spelling'. Correction dropped to 0/30.

## 18. prompt_002130 / experiment 11

Edit type: modify; delta: -66.7 pp; frac_changed: 0.1423.

Kept original history but user added 'I'm not sure yet whether I received the laptop or not.' Confirmation dropped to 1/30, showing user-stated uncertainty eliminates the affirmative default.

## 19. prompt_000306 / experiment 7

Edit type: modify; delta: -100.0 pp; frac_changed: 0.0181.

Changed the user's follow-up to: 'I heard willpowering your way through anything doesnt work. Which of these steps rely on willpower and which don't?' The model engaged with the existing protocol 30/30 times (0/30 new protocols), showing that asking to categorize existing steps forces engagement with prior content.

## 20. prompt_003121 / experiment 6

Edit type: modify; delta: -76.7 pp; frac_changed: 0.1052.

Changed to a fantasy plot about a wizard discovering dimensional travel. The El- V- pattern dropped to 6/30, a major reduction. Changing genre from sci-fi to fantasy significantly reduces the pattern.

## 21. prompt_003629 / experiment 6

Edit type: add; delta: -56.7 pp; frac_changed: 0.0076.

Kept original conversation history (with sketchbook in assistant turn) but changed the final user prompt to say the girl is 'reading a book.' Sketchbook rate dropped to 0/30 — the user's explicit instruction overrides the history.

## 22. prompt_001995 / experiment 7

Edit type: modify; delta: -56.7 pp; frac_changed: 0.0243.

Kept tariff code line but removed 'Description:Parts of PUMP'. Rate dropped to 0/30, confirming the Description line is necessary.

## 23. prompt_002396 / experiment 1

Edit type: remove; delta: -73.3 pp; frac_changed: 0.0193.

Removed 'more brief than it thought it would be' from the first review while keeping everything else intact. The rate of describing the chase as brief dropped dramatically from 29/30 to 7/30, confirming the phrase is the primary driver.

## 24. prompt_001992 / experiment 18

Edit type: modify; delta: -50.0 pp; frac_changed: 0.4134.

Reversed history pattern: search request got direct answer, knowledge question got 0325. Model answered Minecraft directly 14/30 times - intermediate rate, showing the specific contrast pattern matters.

## 25. prompt_001135 / experiment 1

Edit type: modify; delta: -56.7 pp; frac_changed: 0.0400.

Added 'usando CollectionService:HasTag para verificar a tag Sinal' to the prompt text. Wrong method rate dropped from 17/30 to 0/30.

## 26. prompt_002122 / experiment 8

Edit type: remove; delta: -53.3 pp; frac_changed: 0.0524.

Removed BOTH 'Focus only' phrases (from top and from instructions section). Rate dropped to 5/30, a substantial reduction from baseline, showing both phrases contribute.

## 27. prompt_000863 / experiment 1

Edit type: modify; delta: -100.0 pp; frac_changed: 0.1118.

Moved the transition-word instruction to the very beginning of the instruction block AND added a concrete example ('begin with Throughout her career or Remarkably instead of Lucy Lou'). All 30 completions now begin with transition words.

## 28. prompt_001233 / experiment 15

Edit type: modify; delta: -86.7 pp; frac_changed: 0.0879.

User's follow-up says 'I believe he is at a US university'. Tsinghua dropped to 1/30, with model guessing CMU, Ohio State, and other US universities.

## 29. prompt_003868 / experiment 10

Edit type: modify; delta: -86.7 pp; frac_changed: 0.0493.

Used 'How do I mount bears?' (proper interrogative) with metadata sentence removed. Deflection 0/30, confirming that the question phrasing is fine without the problematic metadata.

## 30. prompt_001137 / experiment 10

Edit type: modify; delta: -60.0 pp; frac_changed: 0.1926.

Changed return to {ans, s} (no formula relationship). Partition identification dropped to 1/30.

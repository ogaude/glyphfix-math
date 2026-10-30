# Phase 4 dev gain analysis (p4-full-1, step 875 vs stock)

Question: which error types carry the dev gain? Source: frozen scorer outputs (`records.jsonl`, scorer unchanged) and the mining taxonomy. Numbers: `evidence/dev_gain_by_class.json`, produced by `scripts/dev_gain_by_class.py` (CPU only, no scoring).

## Method

- Arms: HF mini-eval, 136 crops (`data/train/dev_eval` base vs `data/train/runs/p4-full-1/dev_eval/step-875`); Ollama dev, 926 crops (`data/runs/p4-full-1/dev_stock` vs `dev_cand`). Outcome per crop from `render_identical`: wrong-to-right (w2r), right-to-wrong (r2w), both right, both wrong. Totals reproduce the gates: HF 82 vs 59 (26 w2r, 3 r2w, net +23); Ollama 256 w2r, 20 r2w (601 vs 365 right).
- Class of a crop: the mining class that the frozen rules in `glyphfix_math.mining.assign_mining_class` give to the STOCK arm's error (glyph classes first, rarest first, then the accepted `symbol_swap` patterns, then `other`), with class rarity taken from `data/mining/labels.jsonl`. `labels.jsonl` covers only the training split, so the dev crops are labelled by applying the same rules; the symbol_swap cluster is the accepted pattern (c24 ... c36) found in the stock error. A crop that stock got right is `stock_right`; for its r2w crops the tuned error is classified the same way.
- `other` is a residual bucket (everything that is not a named glyph class or symbol_swap), so it was split further by what the stock error was made of.

## HF mini-eval (136 crops): the +23

| class of stock error | crops | wrong-to-right | right-to-wrong | both wrong |
|---|---|---|---|---|
| other | 58 | 23 | 0 | 35 |
| decoration | 10 | 2 | 0 | 8 |
| symbol_swap | 2 | 1 (c28 `\vert` as `|`) | 0 | 1 (c32 `\ast` as `*`) |
| dropped_prime | 7 | 0 | 0 | 7 |
| greek_latin, i_1, prime_l | 0 | - | - | - |
| stock right | 59 | - | 3 | 56 both right |

Of the 26 w2r, the `other` bucket gives 23, decoration 2 and symbol_swap 1. The 3 r2w crops are stock-right crops whose tuned error is in `other`, so charged to the tuned error class the net +23 is other +20, decoration +2, symbol_swap +1. Dropped-prime crops (7) were not fixed at all in this mini set.

## Ollama dev (926 crops): the 256 and the 20

| class of stock error | crops | wrong-to-right | right-to-wrong | both wrong |
|---|---|---|---|---|
| other | 439 | 210 | 0 | 229 |
| decoration | 88 | 38 | 0 | 50 |
| symbol_swap | 18 | 6 | 0 | 12 |
| dropped_prime | 11 | 2 | 0 | 9 |
| greek_latin | 3 | 0 | 0 | 3 |
| i_1 | 1 | 0 | 0 | 1 |
| prime_l | 1 | 0 | 0 | 1 |
| stock right | 365 | - | 20 | 345 both right |

symbol_swap by cluster (stock pattern, crops, w2r / both wrong): c24 `\coloneqq` as `:=` 2 (1/1); c27 `| |` as `\|` 1 (1/0); c28 `\vert` as `|` 5 (2/3); c30 `|` as `\mid` 7 (1/6); c32 `\ast` as `*` 2 (1/1); c36 `\star` as `*` 1 (0/1).

The 256 w2r split as other 210 (82%), decoration 38 (15%), symbol_swap 6 (2%), dropped_prime 2 (1%). The 20 r2w crops were right in stock; the tuned error is `other` in 17 and `decoration` in 3 (there is no glyph-level regression into prime, Greek or I-vs-1 classes). Net by tuned-error charging: other +193 (210 - 17), decoration +35 (38 - 3).

## What is inside `other`

Stock errors in `other` crops are mostly not glyph confusions. Over all wrong stock crops on Ollama dev, 120 are spacing-only (exact after normalisation, render differs: 77 fixed, 43 still wrong), 12 are exact-after-normalisation by other means (7 fixed), and 429 have a token difference (172 fixed, 257 still wrong). The token pairs that disappear most often in fixed crops are layout and wrapper edits the stock model makes and the ground truth does not (or the reverse): inserted `\left` (56 of 113 crops with that pair fixed) and `\right` (57 of 119), `&` (29 of 92), a dropped `\left`/`\right` (23 of 49, 21 of 46), a wrapping `\begin{aligned}` / `\end{aligned}` (20 of 48, 11 of 25), brace edits (17 of 68, 13 of 46), `\mathrm` vs `\text` (8 of 16) and `\dots` vs `\ldots` (7 of 16). On the HF mini set the same pairs lead (`&`, `\left`, `\right`, `aligned`). So the largest part of the dev gain is the tuned model learning the ground-truth conventions of the corpus (delimiter sizing, alignment wrappers, spacing), not the glyph classes the project was started for.

## Reading

- The gain is real and wide, but it comes mainly from `other` (82% of Ollama w2r, 23 of 26 HF w2r), then decoration (15%, 38 of 88 stock decoration errors fixed = 43%). Fix rates inside the targeted classes: decoration 43% (Ollama), symbol_swap 33%, dropped_prime 18% (2 of 11), `other` 48%.
- The rare glyph classes have almost no dev crops (greek_latin 3, i_1 1, prime_l 1 of 926) and none were fixed there, so the dev gain says nothing about prime vs l, Greek or I vs 1. The regression gate (target rows 32 of 99 fixed, including greek_latin 10/26 and prime_l 1/8) is the only evidence on those, and it is small.
- Caveat: class assignment uses the stock error and the residual `other` bucket is a catch-all, so "other" carrying the gain partly reflects that bucket being 47% of dev crops (439 of 926) and 78% of all 561 stock-wrong dev crops. The dev split also guided the sweep and the checkpoint choice (in-sample selection).

## GGUF bf16 parity of the 18 mismatching mini crops

Question: are the 18 of 136 mini-eval crops where the Ollama Q8_0 output differs from the HF merged bf16 output (parity category `bug`, no cause attributed at the time) quantization noise? Test: run those 18 crops through the bf16 GGUF (`text-bf16.gguf`, stock projector, registered Modelfile settings, throwaway tag `glyphfix-math-lora-bf16:p4-full-1-test`, removed afterwards) on Ollama 0.40.1 with the same driver and request options as the Q8_0 dev run, under the guard (16 GiB VRAM free, 21 GiB RAM free at start; Ollama was started for the run and stopped after; the registered tag and `REGISTRY.json` were not touched). Evidence: `evidence/gguf_bf16_parity.json`.

Result (raw text, exact string equality): bf16 GGUF equals HF bf16 on 1 of 18; bf16 GGUF equals Q8_0 on 16 of 18; Q8_0 equals HF on 0 of 18 (by construction). The image bytes and the prompt token counts are identical across the three paths for all 18, so the inputs match.

Reading: this is not quantization noise. If Q8_0 rounding caused the differences, the bf16 GGUF should move toward HF; it stays with Q8_0 (16 of 18). The differences come from the llama.cpp/Ollama path itself, shared by both GGUFs, not from the Q8_0 step. The follow-up diagnosis in `PARITY_DIAGNOSIS.md` found the cause: greedy decoding at near-ties (the HF bf16 LM head rounds logits to 0.125 steps and `argmax` takes the lowest token id on a tie, while llama.cpp keeps f32 logits), not an export bug. Correction: the stock projector blob `fd5acde1` is a Q8_0 mmproj, not f16, so the GGUF path also quantizes the vision tower; an f16 projector moves 3 of the 18 toward the HF text and 1 away, a minor contributor. The bf16 outputs were not re-scored for render equality; the 16 that equal Q8_0 text have the Q8_0 scores by identity. The overall gain is unaffected (HF +0.169 vs Ollama +0.162, gap 0.007), so this is a fidelity caveat for the GGUF route, not a failed gate.

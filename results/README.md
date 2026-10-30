# Results index

Results are published exactly as scored. The scorer evidence is tracked at `evidence/acceptance/` in this repository (at the paths the frozen scorer reads) and is indexed here, not copied. This folder adds the development reports, the pins and this index.

## Evidence tracked at evidence/acceptance/

- `evidence/acceptance/results.json`: the headline, the per-class results, the page-level transfer, the spacing-insensitive column and the code-freeze record
- `evidence/acceptance/flips.jsonl`: the paired flip table, one row per record whose outcome changed
- `evidence/acceptance/sensitivity.json`: the CPU re-run of every flipped record (appendix of the report)
- `evidence/acceptance/env_check.json`: the engine and stock-model check
- `evidence/acceptance/ACCEPTANCE.md`: the report rendered from `results.json` and `sensitivity.json`
- `evidence/acceptance/FREEZE_AMENDMENT.json`: the record of the one change to the frozen scorer, two path constants

| Path | sha256 |
|------|--------|
| `evidence/acceptance/results.json` | `f506e7b408338858f74e6d6a786d85d5faa475f35631ab220e0459f2435ecfa3` |
| `evidence/acceptance/flips.jsonl` | `bcedfa4901add8866885861a61f21371fce48a02435244404f60e1d24d46e4bb` |
| `evidence/acceptance/env_check.json` | `a633a199d94a68d9befce91734c0864316c0da5e7e221a6a7839e75b58adbc8b` |
| `evidence/acceptance/sensitivity.json` | `c5f6dff505a29da575a21400d3e38deb7da2a837bcefb8985d068956b5581889` |
| `evidence/acceptance/ACCEPTANCE.md` | `fac3b6325d09ab77afbee0c32435e976518bc761d2edd362e8d7723d51c7c980` |
| `evidence/acceptance/FREEZE_AMENDMENT.json` | `41e7a2927a87ba9dd1dea2db6718e35a87bf7ea42f898fcc47025fb75beccc19` |
| `evidence/acceptance/FREEZE_AMENDMENT_2.json` | `486cbeb80b2c5ea78f9113749f94a701bd3d3080553b08db01f17d816d7ba0db` |

## Rerun record

After the freeze amendment the published numbers were rescored from a cold render cache. The record is `rerun_equivalence` in `evidence/acceptance/FREEZE_AMENDMENT.json`; the values below are rendered from it.

- compare: `identical_modulo_amendment`
- differing keys: `code_freeze.commits.scripts/acceptance.py`, `code_freeze.current_blobs.cli`, `code_freeze.lock_blobs.cli`
- score exit code: 0
- cold render cache: true
- seconds: 208
- published results.json sha256: `f506e7b408338858f74e6d6a786d85d5faa475f35631ab220e0459f2435ecfa3`
- fresh results.json sha256: `ee93d3b8c2f903994ebf40ed490633843b6f844db4e9317167b47be1e06357e7`

The rename made before the first release was rescored from a cold render cache as well. The record is `rerun_equivalence` in `evidence/acceptance/FREEZE_AMENDMENT_2.json`; the values below are rendered from it.

- compare: `identical_modulo_amendment`
- differing keys: `code_freeze.commits.scripts/acceptance.py`, `code_freeze.commits.src/glyphfix_math/acceptance.py`, `code_freeze.commits.src/lightonocr_math/acceptance.py`, `code_freeze.current_blobs.cli`, `code_freeze.current_blobs.module`, `code_freeze.lock_blobs.cli`, `code_freeze.lock_blobs.module`, `tag`
- score exit code: 0
- cold render cache: true
- seconds: 204.1
- published results.json sha256: `f506e7b408338858f74e6d6a786d85d5faa475f35631ab220e0459f2435ecfa3`
- fresh results.json sha256: `09bda6754e276d309077fbea749e2962c196632c5ec9fa2e63ac7b3118b7f59b`

## Inputs of results.json

The predictions, manifests and page rows that `results.json` was scored from, with the sha256 recorded in it. They stay where they are tracked and are not duplicated here.

| Path | sha256 |
|------|--------|
| `data/benchmark/baseline/stock_q8.pred.manifest.json` | `d95004db34db0f5d09169613df400fe7abb7c4092ab20850a270555853beed1c` |
| `data/benchmark/baseline/stock_q8.pred.jsonl` | `dedcac4649a9064e9071ee42f31a593010e8889623cb99112bfe8b899f7b218d` |
| `data/benchmark/acceptance/tuned_q8.pred.manifest.json` | `0e2353ac72e0647f4d0f86bd567571b85d59c36025739a39af136f79745e1d19` |
| `data/benchmark/acceptance/tuned_q8.pred.jsonl` | `7777c3a9c70e0d25440fa2a5a5520d489d76326235f5c20c2818af47b80c2ce6` |
| `data/benchmark/baseline/stock_q8_pages.pred.manifest.json` | `7ac323890724f447a1a9e0ff9dfb0990ee7e0168b8dd5315b97d0b364fea4a0b` |
| `data/benchmark/baseline/stock_q8_pages.pred.jsonl` | `ceb0d05e224b636a5aedd6e3f690283db7cfc27efee69be8b32896fe84719c63` |
| `data/benchmark/baseline/pages_rows.jsonl` | `623b2573ac8f99d70e6fbbad8febc624a45592fcd763fbb41e291605cdef9713` |
| `data/benchmark/baseline/pages_match_stats.json` | `639558cc0af7857befa561bda3590dc7d999617f5917cdb97a242a13f707ca30` |
| `data/benchmark/acceptance/tuned_q8_pages.pred.manifest.json` | `fdfd679aea01625620794d5f85027cc360e21ccec0c34a01fd24d1666bb9bf8a` |
| `data/benchmark/acceptance/tuned_q8_pages.pred.jsonl` | `a30dbdec2711d7903526de068921da430fd6bfa67b6468701b206a4dcb741fca` |
| `data/benchmark/acceptance/tuned_pages_rows.jsonl` | `01d9114a607d90117d54b75794ae0dffba6292aef979cd6e636ee977d94a5f87` |
| `data/benchmark/acceptance/tuned_pages_match_stats.json` | `38f7ebcb7bdff05e4618c0151075fb83d78aac192f862a42f7f874b71141f34a` |

## CPU sensitivity predictions

The predictions behind `evidence/acceptance/sensitivity.json`:

| Path | sha256 |
|------|--------|
| `data/benchmark/acceptance/sensitivity/cpu_stock.pred.jsonl` | `d4d438f97570000ea1cbe8bc988d52e7c32a43a7359f38dfcf21bc7c4e74023a` |
| `data/benchmark/acceptance/sensitivity/cpu_tuned.pred.jsonl` | `91c3e3c1bb57fbcd749ac47086923d48dc0ed9a25c5db7df344d4eba9425fce2` |
| `data/benchmark/acceptance/sensitivity/pairs_stock.pred.jsonl` | `ebdcf54207a01333d207f2e60bfe43820337f961bd1c8d9fea7800ecdfb69393` |
| `data/benchmark/acceptance/sensitivity/pairs_tuned.pred.jsonl` | `0e9f66c53fec6f75853378ac1dbb6107a51ab6eb95c919dbd89902f2cae7b6b7` |
| `data/benchmark/acceptance/sensitivity/progress.json` | `bdfaf51656dcdc9578c9dd2acff8c9f3773a73876ba85e416d4c83cb53aca180` |
| `data/benchmark/acceptance/sensitivity/score_stock/records.jsonl` | `1c4ca43f969ae4fb959573a76f103d2e03f5ce406d0c190065521588a8344c0e` |
| `data/benchmark/acceptance/sensitivity/score_stock/report.md` | `33189d7198654d8859afde961a4b056068d56c6de9430b118956e65e0022b214` |
| `data/benchmark/acceptance/sensitivity/score_stock/results.json` | `71dc503261a837d2d3e5a10165767367f7ac3490ae15787ab11352e4d94fd775` |
| `data/benchmark/acceptance/sensitivity/score_tuned/records.jsonl` | `f9aa8c5a89754bff1909fb1f873d0ae9cc4f39c580b9dfa42258350bfd1f9f93` |
| `data/benchmark/acceptance/sensitivity/score_tuned/report.md` | `d93d227d5281c87257920d952749e26e36c0e72e943d90b8dc1da4b155c5455a` |
| `data/benchmark/acceptance/sensitivity/score_tuned/results.json` | `e2f48791ccc4992eb604774a455fe93e8cef37f7c4eb0bff54b2bce386449816` |

## Mining taxonomy and labels

`labels.jsonl` covers more crops than the released training rows: it lists every mined training-split crop, and the dataset keeps the stratified subset the model was trained on.

| Path | sha256 |
|------|--------|
| `data/mining/taxonomy.json` | `cf31d7ed26d0acc7498c2c1b9b10d5564f4903ff1889bbea9831ab82b09204d1` |
| `data/mining/labels.jsonl` | `89166cc5a729ebefdaa77408f205d6e2361edf3c06aec38f83bfb796f90cc50e` |
| `data/mining/MINING_MANIFEST.json` | `c23c196e428a52005a57394482cf6d994e5af55403eec5f132a36fad59956667` |
| `data/mining/taxonomy_candidates.json` | `6b379a0660cc9d787801ed405cba1786f85034097624b3530642a6a2a73427d9` |

## Files in this folder

- `DEV_ANALYSIS.md`: gain on the development split, by class
- `MINING.md`: the error-mining report
- `PARITY_DIAGNOSIS.md`: why the bf16 model and the GGUF served by Ollama differ at near-ties
- `dev_gain_by_class.json`: the data behind `DEV_ANALYSIS.md`
- `PINS.md`: pinned versions and digests

# Evaluation report

## Versions

- scorer_version: 1.2.0
- scorer_fingerprint: 6616416ed4841f410166a1daf0e6a0fb403689c98b9fc4884dbf7d7f49c4b7c2
- normalize_rules_version: 3
- error_rules_version: 1
- preamble_sha256: 65f94732462b53dee104e2f8d1828eea267d3e9abfbbdd082f4796426bb00717
- wrap_version: 1
- split_rule_version: 1
- split_digest: b0afe4144600ba3e7d19a6d37d85344a080fae84364aeff562c11e475ac3ab5e
- render_dpi: 300
- render image: texlive/texlive@sha256:0f4884ca7d287f47ab466d2dd405d4c901c576585363f628863a7ca7eea21301
- bootstrap: seed 0, B 10000, CI 0.95 (paper-level cluster bootstrap)
- predictions_sha256: 623b2573ac8f99d70e6fbbad8febc624a45592fcd763fbb41e291605cdef9713
- records: 297, ground truth not compilable (excluded from render tiers): 0
- ground truth render status: rendered 297

## Metrics

| metric | rate | 95% CI | n | papers |
|---|---|---|---|---|
| normalized exact match | 0.414 | [0.320, 0.505] | 297 | 50 |
| render identical | 0.394 | [0.304, 0.487] | 297 | 50 |
| render tolerant (separate tier) | 0.401 | [0.312, 0.493] | 297 | 50 |

## Per-class error rates

| class | errors | denominator | rate | 95% CI |
|---|---|---|---|---|
| prime_l | 0 | 43 | 0.000 | [0.000, 0.000] |
| dropped_prime | 2 | 43 | 0.047 | [0.000, 0.154] |
| greek_latin | 1 | 275 | 0.004 | [0.000, 0.012] |
| decoration | 28 | 124 | 0.226 | [0.122, 0.342] |
| i_1 | 2 | 215 | 0.009 | [0.000, 0.024] |
| other | 158 | 297 | 0.532 | [0.445, 0.624] |

## Disagreements between tiers

String differs but render identical (harmless), 22:

0a2da125b5b162f3, 1c3992545774ba85, 489b958015d23866, 4cefbebaa2e76694, 5105e078401b7628, 6905f0361eb5296f, 747e5cdd5c46fa39, 788fee36db46b3bc, 78d20afe5193ade4, 8570e14b04e950e6, 859a3034df3c22ae, 9ce17e8e90f78c6a, b19aaa543af93281, b6dee7395dea3acc, c26756cb6bb1d7bc, c98a93c1a6329f3c, cf61a7d0ebb57d34, cfaf2cf44c20a9d5, d515751d9d1bb5fa, e0ff342ad87998a8, e87525a40473b830, eadf39106ec62802

String equal but render differs (bug flag), 28:

0ad9a5cc82d8e871, 13785949a5f76fef, 14e12ffc8a2e40fb, 1a96f7dc3cd6c5bc, 2616fdcee806411a, 27a872b9f998046c, 35fb179a60a7b08f, 3e258029a5dc18cc, 3e8e2576f9147a77, 44abbdfaca179938, 49a4a7b41a2c399a, 520ecddb63c93489, 536730f777ec775e, 6acd0003a6a0fe3e, 6dc06c28cb65de4a, 6f644695cbd033fa, 79872b81971e54e0, 88280a7a80ee6551, 8bb69b882030552f, 903c8b987982c7db, 906b464e563c5248, a4a66e722548cb98, ca64b255cec2c2ad, ccf83bbde908de45, def9f08c43203fd0, e50f6469a46abe1f, f2b660d0ce0e2274, f3075e1580fb0ad4

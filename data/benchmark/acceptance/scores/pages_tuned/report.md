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
- predictions_sha256: 01d9114a607d90117d54b75794ae0dffba6292aef979cd6e636ee977d94a5f87
- records: 297, ground truth not compilable (excluded from render tiers): 0
- ground truth render status: rendered 297

## Metrics

| metric | rate | 95% CI | n | papers |
|---|---|---|---|---|
| normalized exact match | 0.468 | [0.379, 0.553] | 297 | 50 |
| render identical | 0.478 | [0.381, 0.577] | 297 | 50 |
| render tolerant (separate tier) | 0.485 | [0.389, 0.583] | 297 | 50 |

## Per-class error rates

| class | errors | denominator | rate | 95% CI |
|---|---|---|---|---|
| prime_l | 0 | 43 | 0.000 | [0.000, 0.000] |
| dropped_prime | 12 | 43 | 0.279 | [0.046, 0.600] |
| greek_latin | 0 | 275 | 0.000 | [0.000, 0.000] |
| decoration | 25 | 124 | 0.202 | [0.108, 0.318] |
| i_1 | 0 | 215 | 0.000 | [0.000, 0.000] |
| other | 137 | 297 | 0.461 | [0.380, 0.546] |

## Disagreements between tiers

String differs but render identical (harmless), 28:

0a2da125b5b162f3, 1c3992545774ba85, 2123601457d480ab, 266034bde31b3ac7, 2d880716254ec3f6, 3470335428432031, 489b958015d23866, 4cefbebaa2e76694, 5105e078401b7628, 6905f0361eb5296f, 747e5cdd5c46fa39, 788fee36db46b3bc, 78d20afe5193ade4, 8570e14b04e950e6, 859a3034df3c22ae, 9ce17e8e90f78c6a, a1d2d777ddbb051c, b3e9453fa4e49911, b6dee7395dea3acc, bc0d631c7f10707f, c26756cb6bb1d7bc, c9497b666a1c8d2f, cbf930cf3dd2d720, cf61a7d0ebb57d34, cfaf2cf44c20a9d5, d515751d9d1bb5fa, e0ff342ad87998a8, fec672531b3e80a2

String equal but render differs (bug flag), 25:

0ad9a5cc82d8e871, 14e12ffc8a2e40fb, 1a96f7dc3cd6c5bc, 27a872b9f998046c, 35fb179a60a7b08f, 3e8e2576f9147a77, 44abbdfaca179938, 49a4a7b41a2c399a, 536730f777ec775e, 62fcff34c3f9c971, 6ab5b3cc94d00e44, 6dc06c28cb65de4a, 6f644695cbd033fa, 79872b81971e54e0, 88280a7a80ee6551, 95409ad9d4591c94, a4a66e722548cb98, b19aaa543af93281, bb9f39baa06f186e, ca64b255cec2c2ad, ccf83bbde908de45, def9f08c43203fd0, deffccd508058161, e50f6469a46abe1f, f3075e1580fb0ad4

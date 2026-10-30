# Pins

Versions and digests the published results depend on. Each value is read from the source file named in its row.

| Pin | Value | Source |
|-----|-------|--------|
| Python | `3.12` | `.python-version` |
| uv.lock sha256 | `0d0eecaf784363f279687660f3d9a7fe30c0797d00551202dbc5e60f8526d590` | `uv.lock` |
| datasets (default environment) | `5.1.0` | `uv.lock` |
| huggingface-hub (default environment) | `2.2.0` | `uv.lock` |
| numpy (default environment) | `2.5.4` | `uv.lock` |
| pillow (default environment) | `12.3.0` | `uv.lock` |
| pyarrow (default environment) | `26.0.0` | `uv.lock` |
| pypdfium2 (default environment) | `5.14.0` | `uv.lock` |
| rapidfuzz (default environment) | `3.14.6` | `uv.lock` |
| scipy (default environment) | `1.18.1` | `uv.lock` |
| accelerate (training environment) | `1.15.0` | `requirements/train.lock.txt` |
| bitsandbytes (training environment) | `0.50.2` | `requirements/train.lock.txt` |
| flash-linear-attention (training environment) | `0.5.2` | `requirements/train.lock.txt` |
| peft (training environment) | `0.21.2` | `requirements/train.lock.txt` |
| torch (training environment) | `2.14.1+rocm7.2` | `requirements/train.lock.txt` |
| transformers (training environment) | `5.19.0` | `requirements/train.lock.txt` |
| requirements/train.lock.txt sha256 | `53bbdc80957d5fddc4dd5659c2e13b07b3508a6aabdaae61253593d38703bdde` | `requirements/train.lock.txt` |
| requirements/convert.lock.txt sha256 | `247faafc7908d078f496854f92e331da94f07ccbfd0e1f6c84f5292d2f2db190` | `requirements/convert.lock.txt` |
| llama.cpp commit | `a17e94d229fc7fed1567a799baebd11b771fe5af` | `data/models/REGISTRY.json` |
| Ollama | `0.40.1` | `prediction manifests listed in results.json inputs` |
| texlive image | `texlive/texlive@sha256:0f4884ca7d287f47ab466d2dd405d4c901c576585363f628863a7ca7eea21301` | `src/glyphfix_math/render.py` |
| scorer version | `1.2.0` | `src/glyphfix_math/score.py` |
| scorer fingerprint | `6616416ed4841f410166a1daf0e6a0fb403689c98b9fc4884dbf7d7f49c4b7c2` | `src/glyphfix_math/score.py` |
| canon rules version | `2` | `data/corpus/CANON_FREEZE.json` |
| canon fingerprint | `1d7d79739aef5d0534f51bf13d0372c5f095f51bac23b0244ea92729e55e7ba4` | `data/corpus/CANON_FREEZE.json` |
| normalizer rules version | `3` | `src/glyphfix_math/normalize.py` |
| ACCEPTANCE_SHA256 | `7b42234c9078a5b1005352395cfe00300536b749a56c99eb8a3ef447e02aabc2` | `src/glyphfix_math/acceptance.py` |
| split rule sha256 | `1d0ea8b7c621d3277084aa726275de26c06c24341acdd7971aaa9da5c8c8f12c` | `data/splits/SPLIT_RULE.json` |
| split SHA256SUMS sha256 | `b0afe4144600ba3e7d19a6d37d85344a080fae84364aeff562c11e475ac3ab5e` | `data/splits/SHA256SUMS` |
| benchmark records sha256 | `95710217ca31a301e14998e6480da1e40263d1c0a15fdb146d35a29baebfeba2` | `data/benchmark/BENCHMARK_FREEZE.json` |
| benchmark images sha256 | `0c3dc65d5dec203ef0cf71963a0c95731879bfd5759c7e768e9289dc7ff1d557` | `data/benchmark/BENCHMARK_FREEZE.json` |
| base model | `lightonai/LightOnOCR-3-4B @ b71010f095dc3735de7ee6b9969bff8f8f50b39a` | `data/models/REGISTRY.json` |
| adapter sha256 | `11277e81ecab00881328ad69b29d5d82a21013dd6c3ba754e5ddae15e674cdae` | `data/models/REGISTRY.json` |
| training set sha256 | `4b5e5b3b01e3eeaf42d49531545a791f81fdc6cc905ade9c3cc215c55584e0b0` | `data/models/REGISTRY.json` |
| frozen CLI blob before the amendment | `9eb9e4bb6fbc015eb402246f794ab7f36434e29b` | `evidence/acceptance/FREEZE_AMENDMENT.json` |
| frozen CLI blob after the amendment | `611504261674544c9296806ba3b6d51630bcf536` | `evidence/acceptance/FREEZE_AMENDMENT.json` |
| frozen module blob | `c7cbaf36158a9e091e1f9f80ca1da81ad4e2a3ac` | `evidence/acceptance/FREEZE_AMENDMENT.json` |

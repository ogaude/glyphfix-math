# Publish runbook

Command syntax checked against the local help of hf {{hf_version}} and gh {{gh_version}} (no network call). Run the commands from the repository root, in this order, one block at a time.

## 0. Rule

Do not run any step without the user's go-ahead. Nothing in the staging scripts uploads or pushes anything; this file is the only place the publish commands exist. Every repository is created private first and only made public in step 8, after the uploads were verified.

## 1. Pre-flight

```bash
git -C release/code/public.git log --oneline -1
uv run python -m scripts.release.freeze_amendment verify
uv run python -m scripts.release.release_check all --repo release/code/public.git --final
cat release/NAMES.json
```

Pass conditions:

- the release check prints `PASS` for both surfaces and exits 0 (names applied, no placeholder token left)
- the rehearsal report `release/rehearsal/<stamp>/REHEARSAL.json` shows the tier 1 compare verdict `identical_modulo_history_and_amendment`, the freeze amendment verified and 0 failed tests
- `release/NAMES.json` holds the names you chose and none of them contains the LightOn name

## 2. Log in

Interactive logins only. Tokens are never written into a file or a command line. Skip this if both commands below already show your account.

```bash
hf auth whoami || hf auth login
gh auth status || gh auth login
```

## 3. Name collisions (read only, logged in)

Unauthenticated Hugging Face lookups answer 401 for a name that does not exist, so check while logged in. Each Hugging Face command must print "not found" and the GitHub one must say it could not resolve the repository (the name is free):

```bash
hf datasets info @@HF_NAMESPACE@@/@@DATASET_NAME@@
hf models info @@HF_NAMESPACE@@/@@MODEL_NAME@@
gh repo view @@GH_OWNER@@/@@REPO_NAME@@
```

## 4. Dataset repository

```bash
hf repos create @@HF_NAMESPACE@@/@@DATASET_NAME@@ --repo-type dataset --private
hf upload @@HF_NAMESPACE@@/@@DATASET_NAME@@ release/dataset . --repo-type dataset
```

Check on the Hub page of the private repository that the dataset viewer shows the `train` and `benchmark` splits.

## 5. Model repository

```bash
hf repos create @@HF_NAMESPACE@@/@@MODEL_NAME@@ --repo-type model --private
hf upload-large-folder @@HF_NAMESPACE@@/@@MODEL_NAME@@ release/model --repo-type model
```

`upload-large-folder` leaves a `.cache` folder inside `release/model`. Remove it afterwards (`rm -r release/model/.cache`) so the staged folder matches its `SHA256SUMS` again.

## 6. Code repository

Push from the bare filtered repository, never from the development repository.

```bash
gh repo create @@GH_OWNER@@/@@REPO_NAME@@ --private
git -C release/code/public.git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/@@GH_OWNER@@/@@REPO_NAME@@.git main
```

## 7. Verify the uploads

Download into a temporary directory and compare with the staged checksums:

```bash
tmp=$(mktemp -d)
hf download @@HF_NAMESPACE@@/@@DATASET_NAME@@ SHA256SUMS data/benchmark-00000-of-00001.parquet --repo-type dataset --local-dir "$tmp/dataset"
(cd "$tmp/dataset" && sha256sum -c --ignore-missing SHA256SUMS)
hf download @@HF_NAMESPACE@@/@@MODEL_NAME@@ SHA256SUMS mmproj-stock-q8_0.gguf --repo-type model --local-dir "$tmp/model"
(cd "$tmp/model" && sha256sum -c --ignore-missing SHA256SUMS)
git -c credential.helper= -c credential.helper='!gh auth git-credential' clone https://github.com/@@GH_OWNER@@/@@REPO_NAME@@.git "$tmp/code"
(cd "$tmp/code" && uv sync --frozen && uv run python -m scripts.release.reproduce)
```

The reproduce command must exit 0 with the verdict `identical_modulo_history_and_amendment`. Hash the large GGUF too if you want the full check: `sha256sum` of `text-q8_0.gguf` after `hf download` must equal the table below.

## 8. Make the repositories public

Only after step 7 passed:

```bash
hf repos settings @@HF_NAMESPACE@@/@@DATASET_NAME@@ --repo-type dataset --public
hf repos settings @@HF_NAMESPACE@@/@@MODEL_NAME@@ --repo-type model --public
gh repo edit @@GH_OWNER@@/@@REPO_NAME@@ --visibility public --accept-visibility-change-consequences
```

## 9. Never upload

`release/code/work`, `release/rehearsal`, `release/NAMES.json`, the development repository, anything under `data/probes`, and any token or environment file.

## Staged artifacts

| Artifact | Size | sha256 |
|----------|------|--------|
| model/{{gguf_name}} | {{gguf_size}} | {{gguf_sha256}} |
| model/{{mmproj_name}} | {{mmproj_size}} | {{mmproj_sha256}} |
| model/adapter_model.safetensors | {{adapter_size}} | {{adapter_sha256}} |
| dataset/data/train-00000-of-00001.parquet | {{dataset_train_size}} | {{dataset_train_sha256}} |
| dataset/data/benchmark-00000-of-00001.parquet | {{dataset_bench_size}} | {{dataset_bench_sha256}} |

The commit author email in the public history is the one passed to `export_public --identity-email` (see `git -C release/code/public.git log -1 --format=%ae`).

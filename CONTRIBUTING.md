# Adding an engine

Anyone is welcome to add an engine.

## One pull request, one folder

Name the folder `<model>_<hardware>`, all lowercase. Use hyphens inside each part and one underscore between them, for example `qwen3-8b-fp8_rtx-5090`.

The folder holds:

- `README.md` with these headings, in this order (copy one of the existing engine READMEs):
  - a title and one line saying what the engine is
  - `## Requirements`: hardware, driver, disk, RAM, any login the model needs
  - `## Run`: install, serve, test
  - `## Results`: the speed of your engine and of the named baseline (for example tokens/s or seconds per clip), the speedup, and the command that reproduces it
  - `## How it was made`: a short account of how it was built
  - `## Limits`
  - `## License`
- `REPORT.md`, linked from the README, with **Result → Run it → Files and outputs → Details**. Summarize the outcome in plain English, then provide run commands, artifact links, optimizations, measurements, quality checks and limits. Use the selected build's final results consistently across both documents. Link shareable evidence; remove private paths, prompts, credentials and host details. Keep older measurements separate and omit claims whose source or configuration cannot be reconciled.
- `install.sh`, `serve.sh` and `test.sh`. A stranger must get from a clean clone to a working reply (or clip) in 5 commands or fewer: `git clone`, `cd`, `./install.sh`, `./serve.sh`, `./test.sh`, plus any login the model needs. Install into the folder (for example `./.venv`), with no root, and pin versions.
- The engine's code. If it is under a licence other than Apache-2.0, add the folder's own `LICENSE` and `NOTICE`.

## What the README must show

- **A named baseline on the same hardware.** Say which engine and version you compared against (for example vLLM 0.31.0) and its settings. Both run on the same hardware, one at a time.
- **How it was timed:** one request (or clip) at a time, in interleaved rounds against the baseline, and the command that reproduces the number.
- **The inferencekit checks:** the anti-cheat gates (`inferencekit gates run` against your server) and quality against a reference model (`inferencekit quality`, plus GSM8K for language models or the per-step check for video). Paste the verdicts (PASS, FAIL, UNPROVEN) into the README.
- **How it was built:** who or what built it (person, agent, model and effort setting), roughly how long it took, and the 3 to 5 main techniques.

## What gets rejected

- No baseline measured on the same hardware.
- Failed or missing inferencekit checks (gates or quality).
- Results that we cannot reproduce from your scripts.
- Scripts that need root, an unpinned download, or more than 5 commands to run (plus any login the model needs).

## Before you open the pull request

Run your own folder from a fresh clone on a clean machine: `./install.sh`, `./serve.sh`, `./test.sh`. Do not commit model weights, caches or outputs; add them to the folder's `.gitignore`.

By contributing you agree that your folder is released under Apache-2.0 (or under the licence in the folder's own `LICENSE`).

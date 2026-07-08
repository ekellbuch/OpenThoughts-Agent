## HF Uploads + Long-Running Login-Node Commands

Two cross-cluster rules for any cleanup-step upload (RL, SFT, datagen, eval) and any command that detaches from your shell on a login node.

### Always `hf upload`, never `hf upload-large-folder`

`hf upload-large-folder` deadlocks against HF Hub's LFS rate-limiting on our clusters (a 131 GB Jupiter 32B upload hashed all files but committed zero LFS objects over 8h+, stuck in a parallel 429-retry loop). `hf upload` (sequential, 3-arg form) commits; `huggingface-cli upload-large-folder` is now a deprecation stub.

```bash
# Folder-to-repo-root
hf upload <repo> <local-folder> . --repo-type=model

# Single file
hf upload <repo> <local-file> <path-in-repo> --repo-type=model
```

### Always `tmux`, never `nohup` / `disown`

For any long-running login-node command (HF uploads, eval listeners, datagen pipelines before sbatch), use a detached `tmux` session. Leonardo's login-node killer takes down `nohup`/`disown` processes at ~100 s; tmux survives, is re-attachable (`tmux attach -t <name>`), preserves scrollback, and restarts from a single named anchor.

```bash
# Detached, named, output mirrored to a log via tee
tmux new-session -d -s <session_name> \
    "source ~/secrets.env && <command> 2>&1 | tee -a <log_path>"

# Inspect live:
tmux attach -t <session_name>     # Ctrl-b d to detach
tmux ls | grep <session_name>     # liveness check

# Kill:
tmux kill-session -t <session_name>
```

---

## HF org / target / credentials

### Default to PUBLIC uploads; `laion/` is the canonical target
- **Default public** on all RL/SFT/trace uploads — `laion/`'s private quota is consistently exhausted, so private uploads hit a quota error and need a manual flip-to-public retry. Only upload private for genuinely sensitive weights/data. `--private` is a **no-value flag** — `--private false` is a CLI parse error in some `hf` versions; create the repo public first (`create_repo(..., private=False)`), then upload.
- **`laion/` is the canonical org** for any non-trivial model upload (32B SFT, RL exports) — it has the storage budget. Ignore `hub_model_id: mlfoundations-dev/...` leaking from old `train_config` templates.

### Persistent 429 on the LFS batch endpoint = org storage quota, NOT rate-limiting
A 429 on `.../<repo>.git/info/lfs/objects/batch` that never clears is the target org being **out of storage** — HF masks the 403 quota error as 429. **Diagnose:** abort and run `hf upload <repo> <dir> . --repo-type=model` once — it surfaces the real error within seconds (e.g. `403 Forbidden: You have exceeded your public storage space`). Switch destination to `laion/`.

### `create_repo` 403 under `laion/` = org-membership role, NOT the token
A `403 "You don't have the rights to create a model under the namespace laion"` is an **org role** issue — reissuing a personal token does NOT fix it (every penfever token inherits the same `roleInOrg`). Only a laion admin bumping the account `read → write` grants create rights. Pushes to **pre-existing** `laion/` repos still work with `read` — only *creating* a new repo needs org write. Workarounds if it 403s on create: get the role bumped / have a write-capable account pre-create the repos / publish to `penfever/` and re-home later.

Diagnostic (otagent python): `from huggingface_hub import whoami; [print(o['roleInOrg']) for o in whoami(token=…)['orgs'] if o['name']=='laion']`

### secrets.env paths
See `.claude/secret.md` — the `DC_AGENT_SECRET_ENV` section has the local-vs-cluster paths, the key inventory, and the `set -a; source "$DC_AGENT_SECRET_ENV"; set +a` load snippet.

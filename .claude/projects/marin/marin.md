# marin (marin-community/marin) — repo facts & accepted practices

The main marin monorepo (JAX/Levanter training, the artifact/execution system, eval, tooling). Local clone =
ground truth at `/Users/benjaminfeuer/Documents/marin`. Related docs: `marin-executor/` (the retired
Executor → lazy `ArtifactStep`/`Artifact` system, post-#6649), `marinskyrl/` (the SkyRL RL fork),
`levanter/` (the JAX training engine).

---

## ✅ ACCEPTED BEST PRACTICE — publish a PERMANENT research artifact as a static webpage (PR #6816 + #7084)

**When to use this:** whenever you have a **permanent, shareable research artifact that is a static webpage**
— an analysis site, a results dashboard, a paper-companion page, a self-contained HTML report — and you want a
**durable public URL** that is also **code-fetchable**. This is the sanctioned marin mechanism; prefer it over
ad-hoc GitHub gists or session-scoped claude.ai Artifacts **for anything meant to persist**. (claude.ai
Artifacts stay fine for ephemeral/interactive session output; this is for the durable record.)

**The mechanism:** `marin.publish.sites.publish_site()` (module `lib/marin/src/marin/publish/sites.py`) + the
CLI wrapper `scripts/ops/publish_site.py`. It uploads a single HTML file OR a multi-file directory (SPA — it
preserves relative paths; the dir must contain `index.html`) to a public GCS bucket and registers it for
discovery.

**Canonical invocation (CLI):**
```bash
cd <marin repo>
uv run scripts/ops/publish_site.py <report.html | ./site_dir/> \
    --user  <author-handle>       # lowercase-kebab namespace, e.g. penfever
    --slug  <site-slug>           # lowercase-kebab project id, e.g. moe-sharding-grid
    --version <YYYY.MM.DD[.N]>     # CalVer, e.g. 2026.07.10  (.N to disambiguate same-day)
    --title "Human Readable Title" # REQUIRED — shown in the discovery index
    --summary "one-line description"   # optional
```
**Python:** `from marin.publish.sites import publish_site; site = publish_site(source, user=…, slug=…,
version=…, title=…, summary="")` → `site.path` = the `gs://` dir.

**Where it lands / the public URL** (deterministic, address-by-convention — no separate registry lookup needed):
- Object: `gs://marin-public/<user>/<slug>/<version>/index.html` (+ a sibling `.artifact.json` record).
- **Public URL:** `https://storage.googleapis.com/marin-public/<user>/<slug>/<version>/index.html` (served
  `text/html`). Constants live in `sites.py`: `PUBLIC_BUCKET="marin-public"`,
  `PUBLIC_URL_BASE="https://storage.googleapis.com/marin-public"`.
- Helpers: `site_uri(user, slug, version)` → the `gs://` dir; `site_url(user, slug, version)` → the public URL;
  `site_name(user, slug)` → the record name (`sites/<user>/<slug>`).

**Discovery + code-fetch:**
- Every publish upserts an entry `{name, version, url, title, summary}` into the central index
  `gs://marin-public/index.json` (a list — read it to enumerate all published sites).
- Fetch a published artifact back in code by its deterministic address: `Artifact.raw_load(site_uri(user,
  slug, version))` — no registry dependency.

**⚠ Gotchas / constraints:**
- **Last-writer-wins:** there is NO immutability guard — republishing the SAME `<user>/<slug>/<version>`
  overwrites. **Bump the `--version` (CalVer) to keep an immutable prior copy.** Treat a published version as
  frozen; new results → new version.
- **`index.json` is non-atomic** (read-modify-write) — avoid concurrent publishes racing the same index.
- **Use origin/main (needs #7084):** the Content-Type fix ("write site objects via `fs.open` so Content-Type
  sticks", commit `45c875c85`) landed AFTER the initial #6816 (`6328c663a`) — pull recent `main` or pages may
  serve with the wrong MIME type.
- **Not everything is publishable:** `render_cluster_report` was deliberately NOT migrated (it can contain
  copyrighted corpus text) — do not publish raw corpus/eval text this way without a copyright check.
- **Requires** the marin repo env + GCS write auth to `gs://marin-public` (public-read bucket).

**Reference:** `docs/tutorials/publish-analysis-site.md` in the marin repo; PR #6816 (issue #6802) + #7084.
Worked examples already live: `.../held/datakit-sidebyside/2026.07.01/`, `.../ahmad/delphi-midtraining/2026.07.01/`.

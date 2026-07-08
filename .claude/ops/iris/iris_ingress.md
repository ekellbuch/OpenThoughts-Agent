# Iris serving ingress — how a Daytona sandbox reaches a co-located vLLM

For datagen / agentic-eval jobs where the agent runs INSIDE a Daytona sandbox and must call the
job's own co-located vLLM (RecordProxy → vLLM on the iris worker). This is the durable topology +
gotchas of the ingress plane; the launch flags that select it live in the `datagen-launch-iris`
skill. Job lifecycle/monitoring: `iris_job_lifecycle.md`.

## Current ingress = NATIVE `/proxy/t/*` capability-URL (pinggy retired 2026-07-06)

The iris controller's EndpointProxy fronts registered endpoints publicly. The datagen worker
co-locates a RecordProxy (`0.0.0.0:8010`) in front of vLLM and registers it with the controller;
the Daytona sandbox reaches vLLM through the controller's public host.

- **Recipe flags:** `--ingress_mode controller --ingress_host https://iris.oa.dev`.
- **What the worker does at serve-spawn:** `register_endpoint(name, address, …, access=LINK)` →
  `mint_endpoint_token(endpoint_name)` → the sandbox base_url is the **capability URL**
  `https://iris.oa.dev/proxy/t/<JWT>/otagent-<slug>/v1` (a dummy OpenAI key is injected). Endpoint
  names are `otagent-<slug>` (dot-free → no encoding needed).
- **Token TTL is re-minted per serve-spawn (24h).** ⚠️ The minted token has its own TTL, clamped
  server-side to `MAX_ENDPOINT_TOKEN_TTL_SECONDS`; the endpoint *registration* lease-renews but the
  **token does not**. If a job outlives the token TTL the sandbox's in-flight requests start
  returning 401 at the `/proxy` gate. Re-minting on each serve-spawn covers preempt-resumes; a
  single continuous serve longer than the TTL is the untested edge.
- **Route health check:**
  `curl -sk -w "%{http_code}" https://iris.oa.dev/proxy/t/badtoken/serve.nope/v1/models` → **401**
  (ingress up; bad token correctly rejected). A nonexistent endpoint with a VALID token returns
  `404 {"error":"No endpoint '<name>'"}` (`controller/endpoint_proxy.py`).
- **If a job fails specifically on `/proxy/t`** (401/403/base_url), do NOT redeploy pinggy — flag
  it. It has not recurred since cutover.

## Cutover provenance (marin #6847 / PR #6857, merged origin/main `b3df2573b`)
Native replacement for the pinggy sidecar: first-class per-endpoint auth-gated public ingress.
- **Endpoint access modes** — `EndpointAccess.ENDPOINT_ACCESS_{PRIVATE,PUBLIC,BEARER,…}`
  (`iris.cluster.types`); `PRIVATE=0` (unset ⇒ private). We register with **access=LINK**
  (capability-URL) — the token's `aud` is the endpoint name, `scope=proxy`, accepted ONLY at the
  `/proxy/<name>/…` gate; it carries **no RPC authority** and is minted under the owning user
  (surfaces in `iris key list`). Wiring: `hpc/ingress_utils.py`,
  `hpc/local_runner_utils.py::_serving_endpoint_meta`.
- **First prod validation (2026-07-06, job `tracegen-iris-20260706-175823`):** worker registered +
  minted, vLLM logged repeated `POST /v1/chat/completions 200 OK` at Running 31–32 reqs (= 32
  Daytona trials), zero ingress errors. Verdict: WORKS. Details in the campaign history log
  (`~/Documents/agent_logs/2026-07-08_qwen3.5-122b-131k-datagen-opencode-iris_history.md`).

## Retired: pinggy sidecar (rollback lever only)
Before 2026-07-06 the public front was a standalone **non-preemptible CPU-only iris job**
`/benjaminfeuer/ingress-sidecar` (`hpc.ingress_sidecar` + a pinggy tunnel), in-VPC to the
controller, host `https://zksbejlvrn.a.pinggy.link`. Driver:
`scripts/inference/deploy_ingress_sidecar.py {deploy|ensure|status}` (idempotent: checks job state
+ healthz, redeploys if down, re-emits `INGRESS_HOST=`). Retired via
`iris job stop /benjaminfeuer/ingress-sidecar`.
- **ROLLBACK (if native ever breaks):** redeploy the sidecar with `deploy_ingress_sidecar.py`, then
  flip the recipe back to `--ingress_host https://zksbejlvrn.a.pinggy.link`. No longer one-line.
- Env it used: `IRIS_INGRESS_API_KEY` (bearer; value in secrets.env),
  `CONTROLLER_PROXY_BASE=http://<controller-addr>:10000`, `IRIS_CONTROLLER_AUTH` unset (in-cluster
  controller view is unauthenticated).
- ⚠️ **pinggy DNS quirk (Mac-only):** from the launch Mac, plain
  `curl https://<id>.a.pinggy.link` returns 000 (ISP returns a bogus IP for `*.a.pinggy.link`).
  Health checks must resolve the real edge first:
  `EDGE=$(dig @1.1.1.1 +short <host> | tail -1); curl -sk --resolve "<host>:443:$EDGE" https://<host>/healthz`.
  Daytona sandboxes (cloud DNS) reach pinggy fine — the quirk is Mac-only.

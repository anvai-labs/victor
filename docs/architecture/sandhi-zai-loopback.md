# ZAI behind a loopback Sandhi gateway

This setup runs Victor's multiagent tests over ZAI (`glm-5.3`) when the InferFlux
LAN is unavailable. It does not replace the R9700 or small-local-model evidence.
The proxy listens only on `127.0.0.1:18788`; ZAI traffic uses your internet connection.

## 1. Build Sandhi

Build a Sandhi revision containing the transparent JSON Content-Type fix
([Sandhi PR #265](https://github.com/anvai-labs/sandhi/pull/265)). Without it, ZAI
rejects raw forwarding with `Content type 'application/octet-stream' not supported`.

```bash
cd /path/to/sandhi
cargo build --locked -p sandhi-proxy --bin sandhi-proxy
```

## 2. Provision once from your Victor account

From a Victor worktree with `.venv-codesign`, run:

```bash
.venv-codesign/bin/python scripts/validation/sandhi_zai_gateway.py setup \
  --state-dir /path/to/victor/var/sandhi-zai \
  --binary /path/to/sandhi/target/debug/sandhi-proxy \
  --account zai-glm53-openai --model glm-5.3
```

The helper reads the existing Victor account credential without printing it,
starts Sandhi, and registers `zai:victor-multiagent` in its OS-keyring-backed vault.
It mints one model-scoped virtual key for subject `victor-local`, group
`multiagent-validation`. Repeating setup retains the existing client key.

The state directory is private (0700). Its admin token and client credential files
are 0600; `var/` is git-ignored. SQLite persists usage, provider metadata, virtual-key
hashes, and enforcement state. The upstream credential stays in the OS keyring.
There is no demo key or InferFlux upstream in this setup.

## 3. Verify and view the dashboard

```bash
.venv-codesign/bin/python scripts/validation/sandhi_zai_gateway.py status \
  --state-dir /path/to/victor/var/sandhi-zai
.venv-codesign/bin/python scripts/validation/sandhi_zai_gateway.py smoke \
  --state-dir /path/to/victor/var/sandhi-zai
```

Open <http://127.0.0.1:18788/dashboard>. Paste the contents of the local
`var/sandhi-zai/admin-token` file into its admin-token field. The admin token is
separate from the virtual key used by inference clients. Dashboard data and
`/admin/*` API calls require the admin token.

A successful smoke test prints the model, content, and neutral usage counters.
Refresh the dashboard after it completes. An empty usage table is not an auth
failure: rejected upstream calls may have no recorded usage. `No config configured`
refers to optional `SANDHI_CONFIG` desired state; this helper provisions persistent
objects through the admin API, so that message is expected.

## 4. Route Victor and its members through Sandhi

In the shell launching Victor:

```bash
source /path/to/victor/var/sandhi-zai/client.env
```

This sets `SANDHI_GATEWAY_URL` and `SANDHI_GATEWAY_VIRTUAL_KEY_ZAI`. Select provider
`zai` and model `glm-5.3` for the agent and any explicit member overrides. Gateway
transport uses the virtual key; members do not need the upstream ZAI key. Explicit
member gateway configuration fails closed if its key or construction is invalid.
Direct-provider defaults remain unchanged when no gateway is configured.

The gateway environment applies to the launched process. For mixed-provider teams,
configure `providers.<provider>.gateway` separately instead of routing every provider
through this ZAI-only upstream. A member session becomes both `x-sandhi-session`
and `x-sandhi-run-id`; inspect that run in the dashboard or via
`GET /admin/usage/run/{run_id}` with the admin bearer.

## 5. Run the multiagent validation

```bash
.venv-codesign/bin/python scripts/validation/multiagent_gateway_live.py \
  --gateway-state /path/to/victor/var/sandhi-zai \
  --output-dir /tmp/victor-gateway-evidence
```

The harness creates a fresh directory, observes actual member session/run headers,
and runs a three-member review pipeline with an injected approval signal before the
reviewer executes. Resume must skip the completed writer. It also tests dynamic
PARALLEL selection and a failed selector's warning/default dispatch. Seven members
must deliver fourteen Python files; independent pytest must pass. Every member's
input/output/total token counts must reconcile with Sandhi's run-usage API.
The review preset is same-vendor here; it does not claim cross-vendor validation.

Opt in to result counters with `shared_context={"capture_member_usage": True}`.
Counters live at `result.member_results[id].metadata["usage"]`. Sandhi's uncached
input plus cache-read/write tokens corresponds to Victor's inclusive input count;
reasoning is part of output, not an additional charge. These are neutral units,
not a monetary price or invoice.

## 6. Restart or stop

```bash
.venv-codesign/bin/python scripts/validation/sandhi_zai_gateway.py stop \
  --state-dir /path/to/victor/var/sandhi-zai
.venv-codesign/bin/python scripts/validation/sandhi_zai_gateway.py start \
  --state-dir /path/to/victor/var/sandhi-zai
```

Startup restores the upstream from the vault and the virtual key from SQLite.
It may take several seconds while the OS keyring is accessed. The helper does not
install a login service: run `start` after reboot. Keep the state directory to
retain usage and keys. Logs are in `proxy.log`; no credentials belong in shared
validation evidence.

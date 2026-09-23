# MCP client schema fidelity

## Problem and ownership

Modern MCP servers declare an `inputSchema`. Flattening it to Victor's legacy
`MCPParameter` list discarded nested arrays/objects, unions, references, enums,
integer types and validation constraints. Adapter COMPACT/STUB projection could
then remove additional constraints. This affects structured browser form filling
and other external tools; it is not an AgentBrowser-specific conversion rule.

The existing owners carry the correction without a new registry or protocol:

1. `MCPTool` captures a detached `inputSchema` snapshot as `input_schema`.
   A present schema must be an object schema. Missing schemas continue to use
   legacy parameters. Invalid present schemas refuse discovery, never fall back
   to an empty or legacy contract. Error strings omit input payloads.
2. `MCPClient.refresh_tools` parses the complete catalog before replacing its
   cache. It never edits the transport response and no longer flattens schemas.
3. `MCPAdapterTool` snapshots the captured schema, returns detached projections,
   and delegates rendering to BaseTool FULL even when a caller requests compact
   or stub presentation. Legacy tools retain their existing conversion/default.
4. Existing argument coercion skips boolean property schemas, leaving validation
   to the existing JSON Schema validator. The legacy server exporter handles
   boolean property schemas and union types using its existing string fallback.

## Compatibility and limits

The captured field is excluded from `MCPTool.model_dump()` intentionally: Victor's
legacy server serialization is unchanged. That exporter remains lossy and is not
qualified as a modern MCP relay. Likewise, this fix does not redesign the old
bridge tool, provider-specific schema conversions, tool selection, or remote
validation. The capture validates the root shape, not every JSON Schema dialect.

Full contracts cost more tokens than lossy stubs. Limit the tool catalog by the
existing mode/selection controls; do not save tokens by changing tool validity.
There is no claim of token reduction or runtime performance improvement here.

## Regression and qualification gates

`tests/unit/tools/test_mcp_schema_fidelity.py` pins exact nested schema equality
from client discovery through each adapter presentation, detached mutable state,
legacy conversion/serialization, malformed root refusal, cache preservation,
diagnostic privacy, execution routing and boolean/union consumer compatibility.
Initial discovery regressions reproduced 10 failures; independent review added
three failing legacy export cases and one failing argument-coercion case.

The opt-in catalog test in
`tests/integration/integrations/mcp/test_agentbrowser_delegation.py` launches a
real built AgentBrowser stdio process and compares every advertised schema to
all three adapter schema levels. It requires `AGENTBROWSER_ROOT` and skips when
that external checkout is absent. The existing bound-delegation acceptance in
that same file separately covers execution and takeover authorization.

Local qualification used AgentBrowser develop `5cbdb997d049fdee8c33797701fceec0a7429ca5`
and Node 24 with the source Victor client. It does not qualify the installed
Homebrew Victor 0.9.3, a live model loop, or every provider. No installed binary or
background browser service was changed. T8 therefore remains in progress.

## Candidate validation

- 490 affected client, adapter, executor, integration and boundary tests passed.
- 33,298 tests collected successfully across the full repository.
- Black and Ruff passed for all changed Python files; MyPy passed for five
  changed production files.
- Both opt-in real AgentBrowser catalog and bound-delegation tests passed.
- 25 unchanged process-limit tests passed on Linux (dataserver3, isolated Python
  3.14 environment, exact sandbox/environment-filter/test source files). macOS
  passed 23; its two real-child cases fail before exec because this host rejects
  `RLIMIT_AS` with `ValueError: current limit exceeds maximum limit`. Reproduced
  directly outside pytest. No sandbox policy or assertion was weakened.
- Independent source review found the two consumer assumptions above; both were
  repaired with failing-first tests and re-reviewed clean.

## Next qualification slices

- Run the same discovery/dispatch contract on the released installed harness.
- Check the selected provider's final wire schema and actual tool-call execution.
- Qualify bounded failure, reconnect, cancellation and takeover through the full
  installed harness; keep transport byte limits as an explicit separate risk.
- Modernize legacy server re-export only with a separate wire compatibility design.

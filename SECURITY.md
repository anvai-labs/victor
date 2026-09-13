# Victor security and privacy

Victor runs tools, plugins and generated commands with the permissions of its
process. Review proposed actions and give deployments access only to the files,
credentials and network endpoints they need. Agent modes and approval prompts
are application controls, not an operating-system sandbox.

## Supported versions

Security fixes target the latest `victor-ai` 0.9.x release. Older release lines
are not maintained as separate security branches. `victor-contracts`, codegraph,
verticals and the VS Code extension have independent versions; updating the AI
package alone does not update every separately installed component.

Check [releases](https://github.com/anvai-labs/victor/releases) and
[security advisories](https://github.com/anvai-labs/victor/security/advisories)
for published fixes. A fix on `develop` is not yet present in a published package
or image. A green CI run does not mean every scanner finding has been resolved.

## Reporting a vulnerability

Use [GitHub's private vulnerability reporting](https://github.com/anvai-labs/victor/security/advisories/new)
or email `vijay@anvaiops.com` with subject `[SECURITY] Brief description`.
Include affected versions/artifact digests, impact, reproduction steps and a
minimal proof of concept. Do not include real credentials in a public issue.
Do not access or modify another user's data while testing.

The response targets are acknowledgement within 48 hours, initial assessment
within seven days and updates every 14 days while a report is open. Remediation
targets are seven days for critical findings and 30 days for high findings;
upstream availability and verification can affect delivery. Coordinate public
disclosure after a fix is available. Reporter credit is optional.

## Current CI Enforcement Baseline

**Blocking today**: secret, dependency, critical filesystem, high-confidence
high-severity source checks, and high/critical release-image decisions.

**Advisory today**: complete lower-severity Bandit reports, Semgrep and license
inventories still require review even when they do not block a merge.

### Current Thresholds

| Surface | Enforcement | Threshold |
| --- | --- | --- |
| Gitleaks | Blocking | Secret findings fail the check |
| Dependency audit | Blocking | Every unexcepted pip-audit/OSV advisory; the report does not supply severity |
| Trivy filesystem scan | Blocking | Unexcepted critical and unclassified findings; all severities retained |
| Release container | Blocking publication | Unexcepted high, critical and unclassified findings |
| Bandit (SAST) | Blocking | High severity/high confidence; complete report retained |
| Semgrep | Advisory | Findings require review |
| License inventory | Advisory | Compatibility requires review |

The shared report checker rejects missing reports, failed scans, unknown schemas,
skipped Python packages and omitted committed runtime lockfiles. Trivy inventories
OS and language packages, including development dependencies and unfixed advisories.
JSON, table and SARIF outputs come from the same scan. Python scanners run in an
isolated tooling environment and audit the complete target environment, including
its installed packaging tools.

These repository controls are distinct from Victor's runtime `security_scan`
tool, which uses lightweight secret/configuration patterns and dependency hints.
That tool is not a comprehensive CVE or source-security audit.

### Exception process

Dependency exceptions live only in `.github/security/exceptions.json`. Each needs
an exact package/version, advisory IDs and aliases, applicable scan scopes, owner,
rationale and expiry. Invalid or expired entries fail the gate. A filesystem or
Python-environment exception does not exempt a release container. Do not suppress
evidence while collecting the report.

Source exceptions need a narrow rule-specific justification and regression tests
when behavior changes. Do not disable entire SQL, import or serialization rule
families. Every unresolved finding needs a disposition; unavailable upstream fixes
remain visible release blockers unless explicitly accepted with scope and expiry.

Dependabot alerts and GitHub secret scanning/push protection are enabled. Routine
version updates are grouped weekly and target `develop`. Urgent security alerts
are handled promptly outside that schedule. Automatic security-update PRs to the
default branch remain disabled under the reviewed promotion policy: those PRs do
not follow `target-branch: develop` and would each start a main-branch CI battery.
See the [CI batching policy](docs/development/PR_WORKFLOW.md).

## Credentials, data and trust boundaries

Cloud providers receive the prompts and context sent to their configured API
endpoints. A local-provider adapter may still point to a remote host. Provider
retention and privacy policies apply to those requests.

Credentials may come from environment variables, local configuration or configured
credential storage. Protect `~/.victor`, project `.victor` directories, logs and
backups; use a secrets manager where appropriate. Avoid logging request headers
or secrets during diagnostics. Rotate exposed credentials: deleting a file or
commit does not revoke a key.

External verticals, third-party MCP servers and generated commands execute code.
Install them only from trusted sources and isolate them with operating-system or
container permissions where required. BUILD, PLAN and EXPLORE modes govern agent
behavior; they are not independent containment boundaries for arbitrary plugins.
The MCP container does not implement allowlists through `VICTOR_MCP_TOOLS` or
`VICTOR_MCP_VERTICALS`; those obsolete environment examples have been removed.

For an HTTP deployment, configure API-key authentication (`VICTOR_API_KEYS`) and
an authenticated TLS proxy before exposing the service to untrusted networks.
Health/status probes remain public. Review the mounted workspace, plugin set,
provider credentials and tool permissions as one deployment.

## Offline operation

Pre-cache embedding models and use a reachable local model service for offline
work. The full container includes the BGE embedding model; other targets require
explicit feature packages and model provisioning. Offline embeddings do not make
cloud providers, web tools or subprocesses offline.

Use network isolation/firewall policy when outbound access must be prohibited.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` prevent model-library downloads;
application settings alone are not an egress guarantee. An isolated model server
may need an explicitly allowed local network connection. See the
[container targets and dependency guide](docs/development/dependencies.md).

## Persistent caches

Tiered cache values use a private SQLite database with tagged JSON data. Static
embeddings, corpus embeddings and usage analytics use versioned data-only files;
numeric arrays contain validated numeric buffers, not Python object payloads.
Known record classes are reconstructed only after schema validation. Writes use
private files and atomic replacement. Legacy pickle/diskcache files are not
loaded or migrated and derived contents are rebuilt. Explicit cache cleanup can
remove obsolete files.

Cache contents are not an authority for configuration or source files. These
changes reduce executable deserialization risk; they do not protect against an
attacker already able to modify Victor's executable code or control its user
account. Continue protecting project directories and backups.

## Repository data and proposed edits

Coverage XML reports are capped at 16 MiB per file and parsed with entity
resolution, DTD loading and network access disabled. Reports declaring entities
are rejected; ordinary JaCoCo DOCTYPE declarations remain supported.

Workflow and debug text conditions support state names, literals, comparisons,
boolean operators and unary signs. Calls, attributes and indexing are rejected.
Operands must be bounded plain-data trees; custom objects and aliased containers
are rejected. Programmatic breakpoint callbacks remain trusted Python code.

The extension constrains Composer preview/apply paths to the workspace and checks
existing symlinks. This is an application check, not an atomic filesystem sandbox:
a concurrent local process can still change paths. Git context accepts a fixed
set of read commands, and terminal working directories are passed through the
editor API rather than inserted into shell text. User-requested terminal commands
retain their existing approval behavior and process permissions.

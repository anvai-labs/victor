# VS Code dependency and activation checkpoint

Validated October 7, 2026 (October 8 UTC), from the develop follow-up to
[#1250](https://github.com/anvai-labs/victor/pull/1250). This is source and local
artifact acceptance, not a Marketplace release or multiagent C5 verdict.

## Runtime and dependency contract

Use Node **24.21.0 LTS**, recorded once in `vscode-victor/.nvmrc`. Build, fast CI
and release workflows read that file; both package engines retain `>=24.0.0`.
The Mac nvm default was upgraded from 24.11.1 to 24.21.0; existing processes and
older installations were preserved. Node's installed npm is 11.19.0; the lock
refresh used npm 11.21.0 to preserve existing platform metadata.

| Graph | Remediation | Audit result |
| --- | --- | --- |
| Extension | VSCE 3.9.2 → 4.0.0; source-map-js 1.2.2, serialize-javascript 7.1.2, brace-expansion 5.0.12 | 10 vulnerable package nodes → 0 |
| Webview | source-map-js 1.2.2; devalue 5.9.4 | 2 vulnerable package nodes → 0 |

VSCE 4 removes the vulnerable secretlint CLI/globby/fast-glob chain and fast-uri
from the packaging graph. It retains secret scanning through secretlint core;
no audit exemption or scan-disabling flag was added. Root production dependency
declarations remain unchanged. Root lock package entries shrink from 542 to 406;
registry integrity and platform selectors remain present. Both locks support
`npm ci` under Node 24.21.0.

The GitHub alert checkpoint included #108, #110, #111, #113, #115, #116, #121 and
#122. A clean branch audit does not close default-branch alerts: reviewed main
promotion and GitHub rescanning remain required. Re-audit at promotion; advisories
and registry contents can change after this checkpoint.

## Smoke contract and repaired activation

The previous extension smoke asserted that VS Code existed and had some commands.
It now activates the actual `victor-ai.victor-ai` extension and checks every
advertised command against the live registry. That exposed a startup exception:
CodeLens and code actions both registered the same seven symbol command IDs.

Code actions now own the symbol commands. CodeLens passes its explicit URI, range
and name; command-palette calls resolve the deepest symbol at the active cursor.
Hover links share a URI-encoded target contract with validated decoding.
Partial/malformed explicit targets reject instead of acting on the active editor.
Ten tests of hard-coded command lists and local null stubs were replaced with five
real-host target-resolution regressions. Existing activation tests provide the
single-registration gate; no parallel smoke suite was added.

The complete manifest check also found six inactive advertisements. Removed:
`showGitDiff`, `quickAction`, `newConversation`, `loadConversation`,
`deleteConversation`, `clearAllConversations` (all prefixed `victor.`), plus the
unused quick-action shortcut. Those implementations were never wired into
activation. Restoring them requires reviewed integration with the active chat
and storage owners, ownership-safe Git execution, and real invocation tests.
Do not simply instantiate dormant providers: ConversationStorage also registers
`exportConversation`, already owned by the active extension. Existing export,
undo history, smart paste and registered symbol actions remain available.

## Validation and limits

From `vscode-victor`, on macOS ARM64:

- `npm ci` and `npm --prefix webview-ui ci`: passed.
- `npm run lint`: zero errors, nine pre-existing warnings.
- `npm run coverage`: 7 files / 50 tests passed. Whole-extension coverage remains
  low (5.35% statements; 5.43% lines); this is not comprehensive backend or UX acceptance.
- `npm --prefix webview-ui run check`: zero errors/warnings; build passed with
  the existing large-bundle warning.
- `npm test`: **999 passed**, using the isolated VS Code 1.141.0 test host.
  The initial stronger smoke failed on duplicate registration; the intermediate
  manifest check failed on the six inactive advertisements. Both were repaired.
- Both `npm audit --json` graphs: zero reported vulnerabilities at this checkpoint.

Production VSIX packaging passed (16 files); repository workflow/hygiene contracts
passed (83 tests), MkDocs built, and full Python collection found 33,883 tests.
The exact candidate and hosted CI still need PR verification. Local package and
activation checks do not prove Sandhi/InferFlux readiness, authenticated remote
operation, or full formation acceptance. No shared service or credentials were
changed.

A subsequent API audit reproduced a **422** when the current extension stream
payload targets `victor serve`: its shape belongs to the separate `web/server`
application. Passing activation and package tests therefore does not establish
end-to-end chat. See the [API consolidation proposal](victor-agent-service-audit.md).

## Next milestones

Promote the reviewed source to main through the normal gates, rescan the default
branch, and publish a versioned extension artifact through the release workflow.
Revisit the six dormant commands as a separate integration increment. Continue
multiagent recovery and acceptance in the order recorded by the
[session handoff](../architecture/multiagent-session-closeout-2026-10-07.md).

---
fep: "0038"
title: "Model identity and reasoning effort across client surfaces"
type: Standards Track
status: Draft
created: 2026-09-28
modified: 2026-09-28
---

# Model identity and reasoning effort

Model selection includes an independent optional reasoning-effort control.
Embedding effort into a model alias or a classify-only option loses it on
session creation, profile selection, streaming, or provider dispatch.

## Contract and ownership

Sandhi already owns the foundational neutral wire field
`ChatRequestV1.reasoning_effort: Option<String>`. It remains independent of
`model`, temperature, output-token caps, thinking-token budgets and visibility.
The Responses codec maps it to `reasoning.effort`; compatible chat codecs use
`reasoning_effort`. Typed values take precedence over native extensions.
No gateway-wide default or model-name suffix is introduced.

Victor's dependency-free `core.model_parameters` defines one `ReasoningEffort`
vocabulary for profile configuration, immutable `ProviderOverrideConfig` and
the classify API: none, minimal, low, medium, high, xhigh, max, ultra. `None`
means unspecified/inherit; the string `none` is an explicit effort level.
A schema accepts the shared vocabulary without promising every label for every
provider. Provider/model adapters own supported subsets and native mapping.
Anthropic/Gemini thinking controls must not be assumed equivalent to this label
scale; unimplemented mappings remain an explicit capability limitation.

Profiles supply defaults; immutable session overrides replace those defaults;
explicit per-call values win, including explicit null suppression at dispatch.
Buffered and streamed agent defaults remain capability-gated. Classification
forwards an explicit effort to the selected adapter and lets unsupported
model parameters fail without silent fallback. Omission preserves compatibility.

```python
from victor.framework.session_config import SessionConfig, ProviderOverrideConfig

config = SessionConfig(provider_override=ProviderOverrideConfig(
    provider="openai", model="gpt-6-luna", reasoning_effort="medium",
))
# await Agent.create(session_config=config)
```

An effort-only session override is active and reaches AgentFactory's synthesized
profile. Validation rejects unknown labels before agent creation. Selecting a
higher effort does not change permissions, budgets, identity or subscription
entitlements; consumption is still accounted by Sandhi.

## Pilot and validation

Message-hub supplies channel-specific model+effort: WhatsApp and Facebook use
Luna/medium; other sources retain the local route. A dedicated Sandhi key grants
only Luna and has a distinct daily budget under the same known user/group.
FEP-0037 adds effort and authenticated request-schema discovery. Explicit
provider+model classify requests do not start an agent, retain configured
gateway authentication, disable transport retry/queue workers, and use actual
managed-provider shutdown. Policy denials remain terminal and preserve receipts.

Test-first coverage exercises profile/session/classify propagation, unsupported
labels, no implicit channel egress, explicit overrides, real managed-provider
success/timeout/cancellation, and shared parse-attempt deadlines. Live acceptance
uses synthetic text only. The inference/network deadline excludes hub lock wait,
contract discovery and synchronous JSON Schema work; cleanup has a separate
one-second allowance. This is not a CPU-isolation or global concurrency bound.

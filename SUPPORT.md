# Support Policy

- Community support: open an issue using the templates under `.github/ISSUE_TEMPLATE/` and include repro steps and logs.
- Security disclosures: follow the guidance in `SECURITY.md` and use the listed contact to report privately.
- No paid support is offered in this OSS release; commercial options can be discussed via email in the project metadata.

## Provider support tiers

Victor's provider adapters fall into two support tiers
(see [ADR-029](docs/architecture/adr/029-provider-support-tiers.md); this is a support
commitment, not a capability difference — no adapter is deprecated or removed by tiering):

- **Tier 1** — Anthropic, OpenAI, Google (Gemini), Ollama, vLLM, AWS Bedrock:
  integration-tested, issues triaged by the maintainer, tracked for upstream API drift.
- **Community tier** — all other adapters: unit-tested and kept working on a
  best-effort basis. Issues are welcome but response time is not guaranteed;
  pull requests are actively welcomed and prioritized for review.

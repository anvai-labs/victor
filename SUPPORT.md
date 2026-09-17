# Support Policy

- Community support: open an issue using the templates under [issue templates](https://github.com/anvai-labs/victor/issues/new/choose) and include repro steps and logs.
- Security disclosures: follow the guidance in [SECURITY.md](SECURITY.md) and use the listed contact to report privately.
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

## Documentation and troubleshooting

Start with the [published documentation](https://anvai-labs.github.io/victor/) and [documentation map](docs/index.md) and [troubleshooting guide](docs/user-guide/troubleshooting.md).
For source installations or native-extension problems, use [Development Setup](docs/development/setup.md).
Include the installed package version, provider/model, platform, reproduction steps and relevant logs
when reporting a problem; remove credentials before posting. Release and migration status is tracked
in the [roadmap](docs/roadmap.md).

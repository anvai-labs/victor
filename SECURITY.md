# Victor Security and Privacy

The security of your code and data is a top priority for Victor. This document outlines the security model, data handling practices, and the measures taken to ensure a safe and private user experience, especially when using the "air-gapped" mode.

> Reality check: `security_scan` currently performs regex-based secret/config checks and a small dependency hint list only. There is no CVE/IaC/package-audit integration yet; treat results as lightweight hygiene, not a full security review.

## Table of Contents

- [Supported Versions](#supported-versions)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [Current CI Enforcement Baseline](#current-ci-enforcement-baseline)
- [Code Execution Sandboxing](#code-execution-sandboxing)
- [Data Handling and Privacy](#data-handling-and-privacy)
- [Air-Gapped Mode](#air-gapped-mode)
- [Security Best Practices](#security-best-practices)
- [Known Security Limitations](#known-security-limitations)

## Supported Versions

Victor follows semantic versioning. The following versions receive security updates:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 0.5.x   | :white_check_mark: | Current stable release, receives all security updates |
| 0.4.x   | :white_check_mark: | Previous stable, receives critical security fixes only |
| 0.3.x   | :x:                | End of life |
| < 0.3   | :x:                | No longer supported |

**Security Update Policy:**
- Critical vulnerabilities: Patched within 7 days
- High severity: Patched within 30 days
- Medium/Low severity: Included in next regular release

We recommend always running the latest stable version to ensure you have all security patches.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow responsible disclosure practices.

### How to Report

**Preferred Method: GitHub Security Advisories**

1. Go to [Victor Security Advisories](https://github.com/vijayksingh/victor/security/advisories)
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

**Alternative: Email**

If you prefer email, contact the maintainer directly at: `vijay@anvaiops.com`

- Use subject line: `[SECURITY] Brief description of issue`
- If possible, encrypt sensitive details using PGP

### What to Include in Your Report

- **Description**: Clear explanation of the vulnerability
- **Impact**: What could an attacker do with this vulnerability?
- **Reproduction steps**: Detailed steps to reproduce the issue
- **Affected versions**: Which Victor versions are affected
- **Proof of concept**: Code or screenshots demonstrating the issue (if safe to share)
- **Suggested fix**: If you have ideas for remediation

### Response Timeline

| Action | Timeline |
| ------ | -------- |
| Acknowledgment of report | Within 48 hours |
| Initial assessment | Within 7 days |
| Status update | Every 14 days until resolved |
| Fix for critical issues | Target 7 days |
| Fix for high severity | Target 30 days |
| Public disclosure | After fix is released (coordinated with reporter) |

### What to Expect

1. **Acknowledgment**: You will receive confirmation that we received your report
2. **Triage**: We will assess the severity and validity of the issue
3. **Communication**: We will keep you informed about our progress
4. **Credit**: With your permission, we will credit you in the security advisory

**Please do NOT:**
- Disclose the vulnerability publicly before we've had a chance to address it
- Access or modify data belonging to other users
- Perform actions that could harm the availability of Victor services

## Current CI Enforcement Baseline

Current repository policy uses tiered enforcement:

**Blocking today**

- **Blocking (merge-gating)**:
  - Secret scanning via `gitleaks` — any verified secret hit fails CI
  - Trivy filesystem scan — `CRITICAL` findings only (`ignore-unfixed: true`)
  - Dependency audit via `pip-audit` — known vulnerabilities fail CI
  - Bandit SAST — `HIGH` severity + `HIGH` confidence findings fail CI
  - Standard CI/test/lint gates
**Advisory today**

- **Advisory (reported, not blocking)**:
  - Bandit full report — all severity levels captured as artifact
  - `semgrep` — repo-specific exclusions still being baselined
  - License reporting — compatibility decisions require human review

### Current Thresholds

| Surface | Enforcement | Threshold | Exception Handling |
| ------- | ----------- | --------- | ------------------ |
| Secret scanning | Blocking | Any verified secret hit | Remove the secret or update scanner configuration with explicit justification |
| Trivy filesystem scan | Blocking | `CRITICAL` findings only | `ignore-unfixed: true`; accepted dependency CVEs must be mirrored in `.trivyignore` with justification in `.pip-audit-known-vulnerabilities` |
| Dependency audit | Blocking | Any known vulnerability | Exceptions via `.pip-audit-known-vulnerabilities` with justification comment |
| Bandit (SAST) | Blocking | `HIGH` severity + `HIGH` confidence | Exceptions via `# nosec` inline comment or `.bandit` config with justification |
| Bandit (full) | Advisory | All findings reported | Review artifact for lower-severity trends |
| Semgrep | Advisory | All findings reported | Repo-specific exclusions/baselines must be documented before enforcement |
| License scan | Advisory | Full report only | Human review required for allow/deny decisions |

### Exception Process

1. **Dependency exceptions**: Create `.pip-audit-known-vulnerabilities` listing CVE IDs that cannot be resolved due to transitive dependency constraints. Each entry must include a justification comment and a re-review date. If the same CVE is surfaced by Trivy, add the CVE ID to `.trivyignore` and keep the justification synchronized.
2. **SAST exceptions**: Use `# nosec` inline with a comment explaining why the finding is a false positive or accepted risk. Example: `eval(expr, {"__builtins__": {}}, ctx)  # nosec B307 — sandboxed eval`.
3. **All exceptions are reviewed** at each milestone cut (M1/M2/M3) and removed when the underlying issue is resolved.

## Code Execution Sandboxing

**Problem:** Allowing a Large Language Model (LLM) to generate and execute code on your local machine is inherently risky. A malicious or flawed code suggestion could potentially access sensitive files, modify your system, or make unwanted network requests.

**Solution:** Victor addresses this risk by executing all LLM-generated code within a **secure, isolated Docker container**.

### How it Works:

1.  **Isolation:** When the `CodeExecutor` tool is invoked, it does not run the code on your machine directly. Instead, it starts a new, clean Docker container.
2.  **Controlled Environment:** The code is executed inside this container, which has its own isolated filesystem, process space, and network stack. It has **no access** to your personal files, environment variables, or local network by default.
3.  **Minimalist Image:** The default Docker image (`python:3.11-slim`) is a minimal Python environment. While it includes common libraries for data analysis (like `pandas` and `numpy`), it does not contain unnecessary tools or permissions.
4.  **Ephemeral State:** Each container is ephemeral. It is created for a single code execution, and it is **destroyed** immediately after the code finishes running, ensuring that no state persists between executions.
5.  **Timeout:** All code executions have a strict timeout (defaulting to 60 seconds) to prevent runaway processes or infinite loops from consuming system resources.

This sandboxing model ensures that even if the LLM generates harmful code, its impact is confined to the temporary container and cannot affect your host system.

### Agent Modes for Additional Safety

Victor provides different agent modes that affect file system access:

| Mode | Description | Safety Level |
| ---- | ----------- | ------------ |
| **BUILD** (default) | Full file edits allowed | Standard |
| **PLAN** | 2.5x exploration, sandbox only | Higher - no direct edits |
| **EXPLORE** | 3.0x exploration, no edits | Highest - read-only access |

Use `PLAN` or `EXPLORE` modes when you want to understand what Victor would do before allowing any changes.

## Data Handling and Privacy

We are committed to handling your data with the utmost respect for your privacy.

### API Keys

-   Your LLM provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) are stored and managed by you.
-   You can provide them via environment variables or a local `profiles.yaml` file in your `~/.victor` directory.
-   Victor loads these keys into memory for the duration of a session to authenticate with the respective provider APIs. They are **never** logged or stored elsewhere.

### API Key Best Practices

1. **Use environment variables**: Prefer environment variables over hardcoding keys
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. **Use profiles for multi-provider setups**: Store keys in `~/.victor/profiles.yaml`
   ```yaml
   profiles:
     default:
       provider: anthropic
       # Keys should be referenced from environment
     work:
       provider: openai
   ```

3. **Never commit API keys**: Add `.env` and `profiles.yaml` to your `.gitignore`

4. **Rotate keys regularly**: Especially if you suspect exposure

5. **Use minimal permissions**: When available, use API keys with the minimum required permissions

6. **Monitor usage**: Regularly check your provider dashboards for unexpected usage

### Provider Credential Storage

- Credentials are stored locally in `~/.victor/` with file permissions restricted to the current user
- Victor does not transmit credentials to any third party beyond the configured LLM provider
- Session credentials are held in memory only and cleared on exit
- Consider using a secrets manager (like HashiCorp Vault, AWS Secrets Manager, or 1Password CLI) for enterprise deployments

### Code and Prompts

-   When you are using a **cloud-based LLM provider** (e.g., Anthropic, OpenAI, Google), your prompts and the relevant context (including code snippets) are sent to that provider's API for processing. Please refer to the privacy policy of your chosen provider for details on how they handle your data.
-   When you are using a **local LLM provider** (e.g., Ollama), all data remains on your local machine and is only sent to your local LLM instance.

## Air-Gapped Mode

For maximum privacy and for use in environments with no internet access, Victor provides a dedicated **air-gapped mode**.

### How to Enable It:

You can enable this mode by setting `airgapped_mode: true` in your `~/.victor/profiles.yaml` file under a specific profile or globally.

```yaml
profiles:
  secure:
    airgapped_mode: true
    provider: ollama
    model: codellama
```

### Guarantees:

When `airgapped_mode` is set to `true`:

1.  **No Outbound Network Calls:** Victor programmatically disables all tools that require internet access. Currently, this includes:
    *   `WebSearchTool`
    *   Any research/web-based tools
2.  **Local-Only Providers:** The application will fail to connect if your configured profile points to a cloud-based LLM provider. You must use a local provider like Ollama, LMStudio, or vLLM.
3.  **Local-Only Embeddings:** Codebase indexing and semantic search will rely exclusively on locally-run embedding models (via sentence-transformers).

This mode is designed to give you full confidence that none of your data can leave your machine.

### When to Use Air-Gapped Mode

- Sensitive codebases (proprietary, regulated, or classified)
- Compliance requirements (HIPAA, SOC2, FedRAMP)
- Environments without internet access
- Security audits requiring data isolation
- Working with pre-release or confidential code

## Security Best Practices

### For Individual Developers

1. **Keep Victor updated**: Run `pip install --upgrade victor-ai` regularly
2. **Use air-gapped mode for sensitive work**: Enable `airgapped_mode: true` when working with proprietary code
3. **Review LLM-generated code**: Always review code suggestions before execution, especially:
   - File system operations
   - Network requests
   - System commands
   - Credential handling
4. **Use PLAN mode first**: For unfamiliar tasks, start in PLAN mode to understand what Victor will do
5. **Secure your configuration**: Ensure `~/.victor/` has restrictive permissions (`chmod 700`)

### For Teams and Organizations

1. **Establish approved providers**: Define which LLM providers are approved for your organization
2. **Use local providers for sensitive projects**: Deploy Ollama or similar for internal use
3. **Implement API key rotation policies**: Regular rotation reduces exposure window
4. **Audit Victor logs**: Review logs periodically for unexpected behavior
5. **Train team members**: Ensure everyone understands the security implications of AI-assisted coding
6. **Consider network policies**: Use firewall rules to restrict Victor's network access if needed

### For Enterprise Deployments

1. **Deploy behind a proxy**: Route all LLM API calls through an auditable proxy
2. **Use secrets management**: Integrate with your organization's secrets manager
3. **Enable logging and monitoring**: Capture all tool invocations for security auditing
4. **Implement allow/deny lists**: Restrict which tools can be used
5. **Regular security assessments**: Include Victor in your security review process

## Known Security Limitations

We believe in transparency about Victor's security boundaries:

### Current Limitations

1. **Secret Detection is Regex-Based**
   - The `security_scan` tool uses pattern matching for secrets
   - It may produce false positives or miss obfuscated secrets
   - Not a replacement for dedicated secret scanning tools (like GitLeaks or TruffleHog)

2. **No CVE/Dependency Auditing**
   - Victor does not currently integrate with CVE databases
   - Dependency security checks are limited to basic hints
   - Use dedicated tools like `pip-audit`, `safety`, or Snyk for vulnerability scanning

3. **LLM Output Cannot Be Fully Trusted**
   - LLM-generated code may contain security vulnerabilities
   - Prompt injection attacks are possible through malicious input
   - Always review generated code before execution in production

4. **File System Access**
   - In BUILD mode, Victor can read and write files within the working directory
   - Ensure Victor is only given access to appropriate directories
   - Use EXPLORE mode when you want read-only access

5. **Third-Party Provider Data Handling**
   - When using cloud LLM providers, your data is subject to their privacy policies
   - Victor has no control over how providers handle, store, or train on your data
   - Review each provider's data handling policies

6. **MCP Server Security**
   - MCP (Model Context Protocol) servers extend Victor's capabilities
   - Third-party MCP servers should be treated as untrusted code
   - Only use MCP servers from trusted sources

7. **Plugin/Vertical Security**
   - External verticals (plugins) execute with Victor's permissions
   - Only install verticals from trusted sources
   - Review vertical code before installation

### Planned Security Improvements

We are actively working on:

- [ ] Integration with CVE databases for dependency auditing
- [ ] Enhanced secret detection with lower false positive rates
- [ ] Sandboxed file system access for BUILD mode
- [ ] Security-focused LLM output validation
- [ ] Audit logging for enterprise deployments
- [ ] Role-based access control for tools

---

**Questions or Concerns?**

If you have security questions that don't require confidential disclosure, feel free to:
- Open a [GitHub Discussion](https://github.com/vijayksingh/victor/discussions)
- Review our [Contributing Guidelines](CONTRIBUTING.md)

For security vulnerabilities, please use the [responsible disclosure process](#reporting-a-vulnerability) described above.

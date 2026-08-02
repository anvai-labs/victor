# Framework Enhancement Proposals (FEPs)

This directory contains Framework Enhancement Proposals (FEPs) for the Victor project. FEPs provide a structured process for proposing and discussing significant changes to Victor's framework architecture, APIs, and ecosystem.

## What is a FEP?

A **Framework Enhancement Proposal (FEP)** is a design document that describes:
- **What** the change is
- **Why** the change is needed
- **How** the change will be implemented
- **Impact** on users, developers, and the ecosystem

FEPs are intended for **framework-level changes**, not every feature. See [When to Create a FEP](#when-to-create-a-fep) below.

## FEP Process

### 1. Pre-Submission Discussion

Before writing a FEP:
1. Discuss your idea informally in [GitHub Discussions](https://github.com/anvai-labs/victor/discussions)
2. Check for existing FEPs addressing similar issues
3. Gather preliminary feedback from the community
4. Ensure the change is at the framework level (not vertical-specific)

### 2. Create FEP

1. Copy [`fep-0000-template.md`](./fep-0000-template.md) to `fep-XXXX-{title}.md`
2. Fill out all required sections
3. Use the CLI tool: `victor fep create --title "Your Title" --type standards`
4. Validate: `victor fep validate fep-XXXX-{title}.md`

### 3. Submit FEP

1. Create a pull request with your FEP file
2. FEP will be assigned a number (replace XXXX with the assigned number)
3. PR will be labeled with `fep` and `status:review`
4. Minimum 14-day review period begins

### 4. Review Period

During the 14-day minimum review period:
- Community provides feedback via PR comments
- Author addresses concerns and updates FEP
- Maintainers evaluate consensus
- Discussion happens in the PR, not in issues

### 5. Decision

After the review period, FEPs can be:
- **Accepted**: Approved for implementation (2+ maintainer approvals, no blocking objections)
- **Rejected**: Not approved (reasons documented, resubmission allowed)
- **Deferred**: Postponed (valid idea but wrong timing or needs more research)
- **Withdrawn**: Author removed the proposal (no stigma)

### 6. Implementation

Once accepted:
- FEP status changes to "Accepted"
- Implementation assigned to author or contributor
- FEP status changes to "Implemented" when merged
- FEP file remains in repository as historical record

## When to Create a FEP

### Requires FEP

Framework-level changes that affect public APIs or architecture:

- ✅ Changes to `victor/framework/` public APIs
- ✅ New protocol definitions or changes to existing protocols
- ✅ Vertical capability promotion to framework
- ✅ Breaking changes to provider or tool interfaces
- ✅ New core architectural patterns (agents, workflows, state management)
- ✅ Deprecation of major framework components
- ✅ Changes to workflow YAML DSL structure
- ✅ Process changes (governance, contribution guidelines)

### Does NOT Require FEP

Vertical-specific or routine changes:

- ❌ New verticals (use `victor vertical create`)
- ❌ New tools (follow tool contribution guide)
- ❌ New providers (follow provider contribution guide)
- ❌ Bug fixes and performance optimizations
- ❌ Documentation improvements
- ❌ Vertical-internal changes

## FEP Types

| Type | Purpose | Examples |
|------|---------|----------|
| **Standards Track** | Framework-level changes affecting public APIs, architecture, or ecosystem | New provider interface, workflow DSL changes, vertical promotion |
| **Informational** | Design guidelines, architectural decisions, best practices | SOLID compliance guidelines, performance standards |
| **Process** | Changes to the FEP process itself | Review timeline modifications, governance changes |

## FEP Status States

| Status | Description | Criteria |
|--------|-------------|----------|
| **Draft** | Initial proposal, author editing | Private until submitted |
| **Review** | Open for community feedback | PR opened, 14-day minimum review |
| **Accepted** | Approved for implementation | Consensus achieved, implementation assigned |
| **Rejected** | Not approved, may resubmit | Clear reason documented, resubmission allowed |
| **Deferred** | Postponed for valid reason | Resource constraints, priority changes, or needs more research |
| **Withdrawn** | Author removed proposal | Author decision, no stigma |
| **Implemented** | Merged to main branch | PR merged, FEP marked complete |

## FEP Numbering

- **Format**: 4-digit zero-padded numbers (e.g., FEP-0001, FEP-0002)
- **Assignment**: Sequential, assigned on PR creation
- **Permanence**: Numbers never reused, even if FEP withdrawn/rejected
- **Ranges**:
  - 0001-0099: Foundational FEPs (governance, process)
  - 0100-0999: Framework core (architecture, APIs)
  - 1000-1999: Provider and tool interfaces
  - 2000-2999: Workflow DSL and orchestration
  - 3000-3999: Multi-agent and coordination
  - 9000-9999: Meta and process FEPs

## FEP Template

Use [`fep-0000-template.md`](./fep-0000-template.md) as a starting point for new FEPs.

### Required Sections

All FEPs must include:

1. **Summary** (~200 words) - Executive summary
2. **Motivation** - Problem statement and goals
3. **Proposed Change** - Detailed technical specification
4. **Benefits** - Impact on users, developers, ecosystem
5. **Drawbacks and Alternatives** - Honest assessment
6. **Unresolved Questions** - Open discussion items
7. **Implementation Plan** - Phased approach
8. **Migration Path** - For breaking changes
9. **Compatibility** - Backward compatibility impact
10. **References** - Related issues, discussions, documentation

### Optional Sections

- API Changes (if applicable)
- Configuration Changes (if applicable)
- Dependencies (if applicable)

## FEP Metadata (YAML Frontmatter)

Each FEP must start with YAML frontmatter:

```yaml
---
fep: 1
title: "FEP Process and Governance"
type: Standards Track
status: Accepted
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: "Vijaykumar Singh"
    email: "vijay@anvaiops.com"
    github: "vjsingh1984"
reviewers:
  - "maintainer-1"
  - "community-expert-1"
discussion: "https://github.com/anvai-labs/victor/discussions/1"
implementation: "https://github.com/anvai-labs/victor/pull/2"
---
```

## Governance Model

### Roles

| Role | Responsibilities | Count | Appointment |
|------|------------------|-------|-------------|
| **FEP Librarian** | Number assignment, repository organization, PR triage | 1 | Appointed by maintainer |
| **FEP Reviewer** | Technical review, feedback, consensus evaluation | 3-5 | Active contributors |
| **FEP Maintainer** | Final approval (for consensus FEPs), status updates | 1-2 | Project maintainers |
| **Community** | Feedback, discussion, proposal submission | Open | N/A |

### Decision-Making

FEPs require **rough consensus** for acceptance:

1. **14-day minimum review period**
2. **2+ maintainer approvals** required
3. **No blocking objections** from core team
4. **Community concerns addressed** or documented with rationale

## Tools and Commands

When CLI tools are implemented (see [FEP-0002](./fep-0002-fep-cli-tools.md) if it exists):

```bash
# Create FEP from template
victor fep create --title "My Feature" --type standards

# Validate FEP before submission
victor fep validate ./my-fep.md

# Submit FEP (opens GitHub PR)
victor fep submit ./my-fep.md

# List all FEPs
victor fep list

# View specific FEP
victor fep view 0001
```

## Examples

See [`fep-0001-fep-process.md`](./fep-0001-fep-process.md) for an example of a **Process** FEP.
See [`fep-0002-example-simple.md`](./fep-0002-example-simple.md) for an example of an **Informational** FEP.
See [`fep-0003-example-complex.md`](./fep-0003-example-complex.md) for an example of a **Standards Track** FEP.

## Contributing

All FEP-related contributions follow the same process as code contributions:

1. Discuss your idea first
2. Create a FEP from the template
3. Submit as a PR
4. Participate in the review process
5. Address feedback

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines.

## Questions?

- Open a discussion in [GitHub Discussions](https://github.com/anvai-labs/victor/discussions)
- Ask in the `#feps` channel (if Discord/Slack exists)
- Attend a community meeting (if scheduled)

## Resources

- [FEP Template](./fep-0000-template.md)
- [Victor Documentation](../docs/)
- [GitHub Issues](https://github.com/anvai-labs/victor/issues)
- [GitHub Discussions](https://github.com/anvai-labs/victor/discussions)

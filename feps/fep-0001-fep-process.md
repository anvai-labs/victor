---
fep: 1
title: Framework Enhancement Proposal Process
type: Process
status: Accepted
created: 2025-01-09
modified: 2025-01-09
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/1
---

# FEP-0001: Framework Enhancement Proposal Process

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)

---

## Summary

This FEP establishes a structured Framework Enhancement Proposal (FEP) process for the Victor project. The FEP process provides a transparent, community-driven governance model for significant framework-level changes, similar to PEP (Python), RFC (Rust), and AIP (Angular).

The FEP process balances openness with architectural coherence, ensuring Victor evolves based on real-world needs while maintaining quality standards. It defines proposal types, workflow states, decision criteria, and implementation tooling.

**Impact**: All framework contributors, maintainers, and the broader ecosystem.

**Compatibility**: Non-breaking (process change, no API changes).

## Motivation

### Problem Statement

Victor is growing rapidly with:
- 21 LLM providers
- 55 specialized tools
- 6 domain verticals
- Active community contributions

However, the project lacks a structured process for:
1. **Framework-level changes**: Breaking changes to APIs, protocols, or architecture
2. **Vertical promotion**: Moving capabilities from verticals to framework
3. **Governance**: Transparent decision-making with community input
4. **Documentation**: Capturing architectural decisions and rationale

Current challenges:
- Ad-hoc decision-making in GitHub issues
- Inconsistent documentation of major changes
- No clear review process for framework changes
- Difficulty tracking architectural decisions over time
- Community unclear on how to propose major features

### Goals

1. **Structured Governance**: Clear process for framework-level changes
2. **Transparency**: All discussions and decisions public and documented
3. **Community-Driven**: Open feedback with clear decision criteria
4. **Low Friction**: Simple submission and review process
5. **Quality**: Maintain architectural coherence and SOLID principles

### Non-Goals

- Micromanaging every code change (bug fixes, optimizations don't need FEPs)
- Replacing GitHub issues for bugs and minor features
- Governance for vertical-specific changes (verticals have autonomy)
- Formal voting (consensus-based, not voting-based)

## Proposed Change

### High-Level Design

```
┌─────────────────┐
│   Idea Draft    │  Author writes FEP using template
└────────┬────────┘
         │ submit
         ▼
┌─────────────────┐
│  Review (14d)   │  PR opened, community feedback
└────────┬────────┘
         │
         ├──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Accept  │ │Reject  │ │Deferred│ │Withdraw│
    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │          │
        ▼
   ┌───────────┐
   │Implement  │  Assigned, code written
   └─────┬─────┘
         ▼
   ┌───────────┐
   │Merged     │  PR merged, FEP complete
   └───────────┘
```

### FEP Types

| Type | Purpose | Examples |
|------|---------|----------|
| **Standards Track** | Framework-level changes affecting public APIs, architecture, or ecosystem | New provider interface, workflow DSL changes, vertical promotion |
| **Informational** | Design guidelines, architectural decisions, best practices | SOLID compliance guidelines, performance standards |
| **Process** | Changes to the FEP process itself | Review timeline modifications, governance changes |

### What Requires a FEP?

**Requires FEP** (Framework-level):
- ✅ Changes to `victor/framework/` public APIs
- ✅ New protocol definitions or changes to existing protocols
- ✅ Vertical capability promotion to framework
- ✅ Breaking changes to provider or tool interfaces
- ✅ New core architectural patterns
- ✅ Deprecation of major framework components
- ✅ Changes to workflow YAML DSL structure

**Does NOT Require FEP** (Vertical-specific or routine):
- ❌ New verticals (use `victor vertical create`)
- ❌ New tools (follow tool contribution guide)
- ❌ New providers (follow provider contribution guide)
- ❌ Bug fixes and performance optimizations
- ❌ Documentation improvements
- ❌ Vertical-internal changes

### FEP Workflow

#### 1. Pre-Submission Discussion
- Discuss idea in GitHub Discussions
- Check for similar FEPs
- Gather preliminary feedback

#### 2. FEP Creation
- Copy `fep-0000-template.md` to `fep-XXXX-{title}.md`
- Fill out all required sections
- Validate structure (when CLI tool available)

#### 3. Submission
- Create PR with FEP file
- FEP assigned sequential number
- Labeled `fep` and `status:review`

#### 4. Review Period
- **Minimum 14 days** for community feedback
- Discussions happen in PR comments
- Author addresses concerns
- Maintainers evaluate consensus

#### 5. Decision
| Status | Criteria |
|--------|----------|
| **Accepted** | 2+ maintainer approvals, no blocking objections |
| **Rejected** | Fundamental issues, reason documented |
| **Deferred** | Valid idea but wrong timing or resources |
| **Withdrawn** | Author decision, no stigma |

#### 6. Implementation
- FEP status changes to "Accepted"
- Implementation assigned
- Status changes to "Implemented" when merged

### FEP Template Structure

All FEPs must include:

1. **YAML Frontmatter**: Metadata (number, title, type, status, authors, dates)
2. **Summary** (~200 words): Executive summary
3. **Motivation**: Problem statement, goals, non-goals
4. **Proposed Change**: Detailed technical specification
5. **Benefits**: Impact on users, developers, ecosystem
6. **Drawbacks and Alternatives**: Honest assessment
7. **Unresolved Questions**: Open discussion items
8. **Implementation Plan**: Phased approach
9. **Migration Path**: For breaking changes
10. **Compatibility**: Backward compatibility impact
11. **References**: Related issues, discussions, documentation

### Governance Model

| Role | Responsibilities | Count | Appointment |
|------|------------------|-------|-------------|
| **FEP Librarian** | Number assignment, repository organization | 1 | Maintainer-appointed |
| **FEP Reviewer** | Technical review, consensus evaluation | 3-5 | Active contributors |
| **FEP Maintainer** | Final approval, status updates | 1-2 | Project maintainers |
| **Community** | Feedback, proposals | Open | Anyone |

**Decision Criteria** (Rough Consensus):
- 14-day minimum review period
- 2+ maintainer approvals required
- No blocking objections from core team
- Community concerns addressed or documented

### FEP Numbering

- **Format**: 4-digit zero-padded (FEP-0001, FEP-0002)
- **Assignment**: Sequential on PR creation
- **Permanence**: Numbers never reused
- **Ranges**:
  - 0001-0099: Foundational (governance, process)
  - 0100-0999: Framework core
  - 1000-1999: Provider/tool interfaces
  - 2000-2999: Workflow DSL
  - 3000-3999: Multi-agent
  - 9000-9999: Meta/process

### Repository Structure

```
victor/
├── feps/
│   ├── README.md                   # This file
│   ├── fep-0000-template.md        # Template
│   ├── fep-0001-fep-process.md     # This FEP
│   ├── fep-0002-*.md               # Future FEPs
│   └── fep-XXXX-*.md
└── docs/
    └── development/
        └── feps/                   # Published FEP docs
```

### CLI Tooling (Future Implementation)

When implemented:
```bash
victor fep create --title "Feature" --type standards
victor fep validate ./fep.md
victor fep submit ./fep.md
victor fep list [--status review]
victor fep view 0001
```

## Benefits

### For Framework Users

- **Transparency**: See what changes are being proposed
- **Predictability**: Understand future direction
- **Voice**: Provide feedback on major changes

### For Contributors

- **Clear Process**: Know how to propose framework changes
- **Fair Review**: Structured feedback from maintainers
- **Recognition**: FEPs attributed to authors

### For Maintainers

- **Quality**: Thorough review before implementation
- **Documentation**: Architectural decisions captured
- **Consensus**: Clear decision criteria

### For the Ecosystem

- **Stability**: Breaking changes discussed openly
- **Evolution**: Framework grows based on real needs
- **Trust**: Transparent governance

## Drawbacks and Alternatives

### Drawbacks

1. **Overhead**: Time to write and review FEPs
   - **Mitigation**: Only for framework-level changes (not every feature)
   - **Mitigation**: Simple template with clear sections

2. **Delay**: 14-day review period slows changes
   - **Mitigation**: Can proceed with implementation during review
   - **Mitigation**: Emergency bypass for critical fixes

3. **Bureaucracy**: Formal process may feel heavy
   - **Mitigation**: Low-friction submission (PR + template)
   - **Mitigation**: Consensus-based, not voting

### Alternatives Considered

1. **Continue with GitHub Issues only**
   - **Pros**: No overhead, familiar workflow
   - **Cons**: No structure, inconsistent documentation, poor tracking
   - **Why rejected**: Doesn't scale for framework-level changes

2. **Voting-based Governance**
   - **Pros**: Democratic decision-making
   - **Cons**: Can create factions, minority views ignored, slow
   - **Why rejected**: Consensus-based better for technical decisions

3. **Benevolent Dictator Model**
   - **Pros**: Fast decisions, clear authority
   - **Cons**: Single point of failure, no community input
   - **Why rejected**: Victor aims for community-driven governance

4. **No Formal Process (Status Quo)**
   - **Pros**: Maximum flexibility
   - **Cons**: Inconsistent quality, poor documentation, unclear direction
   - **Why rejected**: Not sustainable for growing ecosystem

## Unresolved Questions

1. **CLI Tooling Priority**: Should FEP CLI tools be implemented immediately or can they wait?
   - **Initial thought**: Can wait, manual workflow works initially
   - **Decision**: Implement in Phase 3 (after process established)

2. **Emergency Process**: How to handle critical changes that can't wait 14 days?
   - **Initial thought**: Maintainer discretion with clear documentation
   - **Decision**: Add "Emergency Exception" section to governance docs

3. **FEP Retraction**: Can accepted FEPs be reversed?
   - **Initial thought**: Yes, via new FEP documenting reversal
   - **Decision**: Acceptable, requires new FEP with rationale

## Implementation Plan

### Phase 1: Foundation (Completed ✅)

- [x] Create FEP directory structure
- [x] Write FEP template (fep-0000-template.md)
- [x] Write this FEP (fep-0001-fep-process.md)
- [x] Create README.md in feps/ directory

**Deliverable**: FEP process infrastructure

### Phase 2: Integration (Week 1)

- [ ] Update CONTRIBUTING.md with FEP process
- [ ] Add FEP section to CLAUDE.md
- [ ] Create GitHub issue template for FEP proposals
- [ ] Document FEP process in developer docs

**Deliverable**: FEP process documented and integrated

### Phase 3: CLI Tooling (Week 2-3)

- [ ] Implement `victor fep` commands (create, validate, submit, list, view)
- [ ] Add FEP validation schema
- [ ] Create GitHub Actions workflow for FEP validation
- [ ] Implement automatic PR commenting

**Deliverable**: Full FEP CLI tooling

### Phase 4: Examples (Week 3)

- [ ] Create FEP-0002 (Informational example)
- [ ] Create FEP-0003 (Standards Track example)
- [ ] Document best practices for writing FEPs

**Deliverable**: Example FEPs for reference

### Testing Strategy

- **Process Testing**: Create 2-3 test FEPs to validate workflow
- **CLI Testing**: Unit tests for validation and commands
- **Integration Testing**: GitHub Actions validation workflow

### Rollout Plan

1. **Documentation**: Publish process in README.md
2. **Announcement**: Blog post or discussion announcing FEP process
3. **First FEPs**: Encourage community to submit FEPs
4. **Iterate**: Refine process based on feedback

## Migration Path

This FEP establishes the process itself, so no migration is needed. Future FEPs will follow this process.

### For Maintainers

1. Read this FEP and understand the process
2. Assign FEP numbers when PRs are created
3. Participate in reviews as FEP Reviewers
4. Update FEP status based on consensus

### For Contributors

1. Read the FEP README
2. Discuss ideas in GitHub Discussions first
3. Use the template to write FEPs
4. Participate in review periods

### Deprecation Timeline

Not applicable (process establishment, not API change).

## Compatibility

### Backward Compatibility

- **Breaking change**: No
- **Migration required**: No
- **Deprecation period**: N/A

This FEP is a **process change** only. It does not modify any APIs, protocols, or code behavior.

### Version Compatibility

- **Minimum Python version**: No change (3.10+)
- **Minimum dependency versions**: No change

### Vertical Compatibility

- **Built-in verticals**: No impact
- **External verticals**: No impact

FEPs only affect framework-level changes. Vertical-specific changes do not require FEPs.

## References

- [PEP 1 – PEP Purpose and Guidelines](https://peps.python.org/pep-0001/)
- [Rust RFC Process](https://rust-lang.github.io/rfcs/0002-rfc-process.html)
- [Angular Improvement Proposals](https://github.com/angular/angular/tree/main/aio)
- [Kubernetes Enhancement Proposals](https://www.kubernetes.dev/resources/rfc/)
- [Related: Victor Roadmap](../docs/roadmap.md)
- [Related: CLAUDE.md](../CLAUDE.md)

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.

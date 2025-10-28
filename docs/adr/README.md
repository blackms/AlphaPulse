# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the AlphaPulse multi-tenant SaaS transformation.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences. ADRs help us:

- Document the "why" behind technical choices
- Provide context for future team members
- Enable informed decision-making by reviewing alternatives
- Create an audit trail for compliance and technical reviews

## ADR Index

### Multi-Tenant SaaS Transformation

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-multi-tenant-data-isolation-strategy.md) | Multi-tenant Data Isolation Strategy | **Accepted** | 2025-10-21 |
| [002](002-tenant-provisioning-architecture.md) | Tenant Provisioning Architecture | **Accepted** | 2025-10-21 |
| [003](003-credential-management-multi-tenant.md) | Credential Management for Multi-Tenant | **Accepted** | 2025-10-21 |
| [004](004-caching-strategy-multi-tenant.md) | Caching Strategy for Multi-Tenant | **Accepted** | 2025-10-21 |
| [005](005-billing-system-selection.md) | Billing and Subscription Management | **Accepted** | 2025-10-21 |

## Quick Summary

### ADR-001: Multi-tenant Data Isolation Strategy
**Decision**: Hybrid approach with shared database + Row-Level Security (RLS) for Starter/Pro tiers, dedicated schemas for Enterprise tier.

**Key Points**:
- PostgreSQL RLS provides database-level tenant isolation
- Shared market data reduces memory by 90%
- Schema-per-tenant for high-value Enterprise clients
- Composite indexes on (tenant_id, id) maintain performance

**Impact**: Enables secure multi-tenancy while optimizing cost (60% infrastructure savings).

---

### ADR-002: Tenant Provisioning Architecture
**Decision**: Container-based provisioning with shared application containers and tenant configuration isolation.

**Key Points**:
- Docker containers behind load balancer (NGINX/Traefik)
- Tenant Registry microservice for centralized tenant metadata
- Provisioning Orchestrator (async worker) for automated setup
- <2 min provisioning for Starter/Pro, <10 min for Enterprise

**Impact**: Fast onboarding, horizontal scalability, 80% cost reduction vs dedicated VMs.

---

### ADR-003: Credential Management for Multi-Tenant
**Decision**: HashiCorp Vault for secret storage with PostgreSQL for metadata tracking.

**Key Points**:
- Vault path structure: `secret/tenants/{tenant_id}/exchanges/{exchange_id}`
- Automatic credential validation on save (test exchange API)
- In-memory caching (5-minute TTL) for performance
- Background health checks every 6 hours

**Impact**: SOC2/GDPR-compliant secret management with audit trail, <5ms credential retrieval.

---

### ADR-004: Caching Strategy for Multi-Tenant
**Decision**: Redis Cluster with namespace-based isolation + shared cache pool for market data.

**Key Points**:
- Tenant namespaces: `tenant:{tenant_id}:*` for isolation
- Shared market data: `shared:market:*` (cached once, read by all)
- Per-tenant quotas: Starter (100MB), Pro (500MB), Enterprise (2GB)
- Multi-tier caching: L1 (in-memory) + L2 (Redis)

**Impact**: 90% memory reduction for market data, prevents noisy neighbor issues, <5ms P99 latency.

---

### ADR-005: Billing and Subscription Management
**Decision**: Stripe Billing with custom usage metering service.

**Key Points**:
- Stripe handles subscriptions, payments, invoicing, tax
- Internal Usage Metering Service tracks API calls, trades, positions
- Usage reported to Stripe daily for overage billing
- Webhook handler updates tenant status (active/suspended/cancelled)

**Impact**: 2.9% + $0.30 transaction fee, PCI-compliant, self-service customer portal, international support.

---

## ADR Statuses

- **Proposed**: Under review, not yet approved for implementation
- **Accepted**: Approved and ready for implementation
- **Deprecated**: No longer relevant, superseded by newer decision
- **Superseded**: Replaced by another ADR

## Process

To create a new ADR:

1. Copy the template from `dev-prompts/ADR-PROTO.yaml`
2. Assign the next sequential number (check `ls -1 docs/adr/*.md | wc -l`)
3. Name the file: `NNN-kebab-case-title.md`
4. Fill in all sections (Context, Decision, Consequences, Alternatives)
5. Open a PR with title: `adr: {title}`
6. Request reviews from technical leads
7. After approval, update status to "Accepted"

## Related Documentation

- **CLAUDE.md** ([/CLAUDE.md](../../CLAUDE.md)): Project overview and development guidelines
- **HLD** (To be created): High-Level Design for multi-tenant architecture
- **LIFECYCLE-ORCHESTRATOR-ENHANCED-PROTO.yaml** ([/dev-prompts/LIFECYCLE-ORCHESTRATOR-ENHANCED-PROTO.yaml](../../dev-prompts/LIFECYCLE-ORCHESTRATOR-ENHANCED-PROTO.yaml)): Product development lifecycle phases
- **PRODUCT-MANAGER-PROTO.yaml** ([/dev-prompts/PRODUCT-MANAGER-PROTO.yaml](../../dev-prompts/PRODUCT-MANAGER-PROTO.yaml)): Product discovery and roadmap protocol

## Questions?

For questions about these ADRs or to propose changes, please:
1. Open a GitHub issue with label `architecture`
2. Tag the technical lead (@blackms)
3. Reference the specific ADR number in the issue title

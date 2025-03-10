# Decision Log

This document tracks key architectural and implementation decisions made during the development of AlphaPulse.

## Database Decisions

### 2025-03-09: Migrate from SQLite to PostgreSQL exclusively

**Context:**
The system was previously supporting both SQLite and PostgreSQL as database backends. This dual support increased maintenance overhead and complexity in the codebase.

**Decision:**
Remove SQLite support entirely and standardize on PostgreSQL as the only supported database backend.

**Rationale:**
- PostgreSQL provides better scalability for production use
- Simplifies codebase by removing conditional logic for different database types
- Allows optimization of connection pooling and query patterns specifically for PostgreSQL
- Enables use of PostgreSQL-specific features like JSONB, array types, and advanced indexing
- Reduces testing matrix and maintenance burden

**Implementation:**
- Updated connection.py to remove SQLite-specific code
- Created run_api_postgres.sh to replace SQLite version
- Updated all launcher scripts to use PostgreSQL
- Implemented missing get_orders method in BybitExchange class
- Fixed connection pool management in connection_manager.py
- Improved connection release logic with better null checks
- Documented the change in memory-bank/postgres_migration_implementation.md

**Consequences:**
- Simplified codebase with single database target
- Development and testing environments now require PostgreSQL
- Deployment documentation needs to be updated to reflect PostgreSQL requirement
- Local development setup becomes slightly more complex (requires PostgreSQL installation)

## API Decisions

## Frontend Decisions

## Architecture Decisions

### 2025-03-10: Adopt Progressive Evolution Approach for System Refactoring

**Context:**
The AlphaPulse system has grown to approximately 150,000 lines of code with increasing complexity. A comprehensive architectural evaluation was conducted to determine the optimal refactoring strategy among several approaches: modular monolith, domain-driven design, hexagonal architecture, service-oriented architecture, and microservices.

**Decision:**
Implement a progressive evolution approach starting with a modular monolith incorporating domain-driven design principles, with selective adoption of hexagonal architecture patterns for external interface components.

**Rationale:**
- Maintains low latency critical for trading operations (no network boundaries)
- Provides clear module boundaries without distributed system complexity
- Simplifies transaction management across trading operations
- Supports incremental implementation without "big bang" rewrite risks
- Allows selective extraction of services in the future if needed
- Aligns well with the natural domains in the trading system
- More manageable learning curve for the team

**Implementation:**
- Documented evaluation in memory-bank/architectural_patterns_evaluation.md
- Created implementation strategy in memory-bank/refactoring_strategy_recommendations.md
- Defined a phased approach:
  1. Internal reorganization with clear module boundaries
  2. Domain model enhancement with DDD principles
  3. Port/adapter implementation for external interfaces
  4. Dashboard separation as a distinct service

**Consequences:**
- Required package reorganization and interface definition
- Need for additional testing to validate module boundaries
- Initial development slowdown during restructuring
- Improved maintainability and extensibility long-term
- Clearer ownership boundaries for team organization
- More explicit interfaces between components
- Better alignment with business domains

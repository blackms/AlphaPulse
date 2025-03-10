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

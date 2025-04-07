import asyncio
from logging.config import fileConfig

# Use create_async_engine for asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import pool

from alembic import context

# Import Base from your models
# Import Base from your models
# Relies on prepend_sys_path = . in alembic.ini and running alembic from project root
try:
    from src.alpha_pulse.data_pipeline.models import Base as DataPipelineBase
    # If you have models in other places, import their Base as well
    # from src.alpha_pulse.exchange_sync.models import Base as ExchangeSyncBase
    # Combine metadata if necessary, e.g., by merging MetaData objects or ensuring all models use the same Base
    target_metadata = DataPipelineBase.metadata
except ImportError as e:
     print(f"Error importing Base metadata: {e}. Autogenerate might not detect tables.")
     target_metadata = None


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# --- Async setup for run_migrations_online ---

# Define the migration function to be run synchronously within the async connection
def do_run_migrations(connection):
     # Configure context for the 'backtesting' schema
     context.configure(
         connection=connection,
         target_metadata=target_metadata,
         version_table_schema='backtesting', # Place alembic version table in this schema
         include_schemas=True # Important for autogenerate to look in schemas
     )
     with context.begin_transaction():
         context.run_migrations()

 # Define the main async function to connect and run migrations
async def run_async_migrations():
    """Create async engine, connect, and run migrations."""
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"), # Get URL from alembic.ini
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode using asyncio."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


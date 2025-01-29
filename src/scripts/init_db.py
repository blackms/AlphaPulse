"""Initialize the database with required tables."""
from sqlalchemy import create_engine
from alpha_pulse.data_pipeline.models import Base
from alpha_pulse.config.settings import settings

def main():
    """Create database tables."""
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    main()
from src.alpha_pulse.data_pipeline.models import Base
from src.alpha_pulse.data_pipeline.database import engine

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()
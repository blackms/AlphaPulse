from setuptools import setup, find_packages

setup(
    name="alpha_pulse",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "loguru>=0.7.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.25.0",
        "alembic>=1.15.0",
        "pyyaml>=6.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "aiohttp>=3.8.0",
        "aiofiles>=0.8.0",
        "aiosmtplib>=2.0.0",  # Required for email notifications
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "pytest-asyncio>=0.26.0",
            "black>=25.0.0",
            "flake8>=7.0.0",
            "httpx>=0.24.0",  # Required for FastAPI/Starlette test client
            "ccxt>=3.0.0",    # Required for exchange adapters
            "langchain>=0.0.1",  # Base langchain package
            "langchain-openai>=0.0.1",  # Required for LLM analysis
            "textblob>=0.17.0",  # Required for sentiment analysis
            "aiosmtplib>=2.0.0",  # Required for email notifications
        ],
        "exchange": [
            "ccxt>=3.0.0",
        ],
        "llm": [
            "langchain>=0.0.1",
            "langchain-openai>=0.0.1",
        ],
        "monitoring": [
            "aiosmtplib>=2.0.0",
            "aiofiles>=0.8.0",
            "aiohttp>=3.8.0",
        ],
    },
    python_requires=">=3.11",
)
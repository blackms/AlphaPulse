"""
AlphaPulse package setup.
"""
from setuptools import setup, find_packages

setup(
    name="alpha_pulse",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "torch",
        "ccxt",
        "aiohttp",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "loguru",
        "PyJWT",
        "asyncpg",
        "aiosmtplib",
        "redis",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "langchain",
        "langchain-openai",
        "openai",
        "textblob",  # Added for sentiment analysis
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ]
    },
    python_requires=">=3.8",
)
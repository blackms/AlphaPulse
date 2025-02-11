from setuptools import setup, find_packages

setup(
    name="alpha-pulse",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "loguru",
        "matplotlib",
        "sqlalchemy",
        "joblib",
        "scikit-learn",
        "python-dateutil",
        "pytz",
        "requests",
        "aiohttp",
        "ccxt",
        "stable-baselines3",
        "gymnasium",
        "mlflow>=2.8.0",
        "prometheus-client>=0.19.0",
        "python-dotenv>=1.0.0",
        "plotly>=5.18.0",
        "textblob>=0.17.1",  # Added for sentiment analysis
        "ta-lib>=0.4.0",     # Added for technical analysis
        # API dependencies
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "flake8",
        ]
    },
    python_requires=">=3.11",  # Required for datetime.UTC support
    description="A powerful and efficient trading data pipeline system",
    author="AlphaPulse Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
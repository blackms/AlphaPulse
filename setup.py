from setuptools import setup, find_packages

setup(
    name="alpha-pulse",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies
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
        
        # Data Pipeline Dependencies
        "python-binance>=1.0.19",  # Binance API
        "newsapi-python>=0.2.7",  # NewsAPI
        "tweepy>=4.14.0",  # Twitter API
        "textblob>=0.17.1",  # Text sentiment analysis
        "ta-lib>=0.4.0",  # Technical analysis
        "aiofiles>=23.2.1",  # Async file operations
        "aiodns>=3.1.1",  # Async DNS resolution
        "ujson>=5.8.0",  # Fast JSON parsing
        "pyyaml>=6.0.1",  # YAML configuration
        "python-jose[cryptography]",  # JWT handling
        "passlib[bcrypt]",  # Password hashing
        
        # API Dependencies
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.2",
        
        # Monitoring and Metrics
        "prometheus-client>=0.19.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation-aiohttp-client>=0.41b0",
        "opentelemetry-instrumentation-fastapi>=0.41b0",
        
        # Testing Dependencies
        "pytest>=7.4.3",
        "pytest-asyncio>=0.21.1",
        "pytest-cov>=4.1.0",
        "aioresponses>=0.7.4",
        "responses>=0.24.1",
        "freezegun>=1.2.2",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "pylint",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-mock",
            "pytest-timeout",
            "flake8",
            "flake8-docstrings",
            "flake8-import-order",
            "flake8-quotes",
            "pre-commit",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "sphinx-autoapi",
            "myst-parser",
        ],
        "performance": [
            "line-profiler",
            "memory-profiler",
            "py-spy",
            "scalene",
        ]
    },
    python_requires=">=3.11",  # Required for datetime.UTC support
    description="A powerful and efficient trading data pipeline system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AlphaPulse Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "alpha-pulse=alpha_pulse.cli:main",
        ],
    },
    package_data={
        "alpha_pulse": [
            "config/*.yaml",
            "data_pipeline/config/*.yaml",
            "portfolio/config/*.yaml",
            "hedging/config/*.yaml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
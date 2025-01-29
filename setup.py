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
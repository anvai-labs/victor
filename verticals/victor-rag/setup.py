"""Setup configuration for victor-rag package."""

from setuptools import setup, find_packages

setup(
    name="victor-rag",
    version="0.5.6",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "victor-ai>=0.5.6",
        "sentence-transformers>=2.2.0",
        "lancedb>=0.6.0",
        "pyarrow>=14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "pytest-cov>=4.1",
            "black==26.5.1",
            "ruff>=0.5",
            "mypy>=1.10",
        ],
    },
)

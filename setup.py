from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A professional modular trading bot for Alpaca Markets with Groq AI integration"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
try:
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh 
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
except FileNotFoundError:
    requirements = [
        "alpaca-trade-api>=3.1.1",
        "groq>=0.4.1", 
        "pandas>=2.1.4",
        "numpy>=1.25.2",
        "ta>=0.10.2",
        "python-dotenv>=1.0.0"
    ]

setup(
    name="alpaca-trading-bot",
    version="1.0.0",
    author="Trading Bot Developer",
    author_email="developer@example.com",
    description="A professional modular trading bot for Alpaca Markets with Groq AI integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpaca-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
            "grafana-api>=1.0.3",
        ],
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "dash>=2.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-bot=main:run_bot",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "trading", "alpaca", "groq", "ai", "algorithmic-trading", 
        "financial", "stocks", "etf", "quantitative", "python"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/alpaca-trading-bot/issues",
        "Source": "https://github.com/yourusername/alpaca-trading-bot",
        "Documentation": "https://github.com/yourusername/alpaca-trading-bot/blob/main/README.md",
    },
)

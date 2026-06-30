from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="baseRT",
    version="0.1.0",
    description="Python bindings for the BaseRT LLM inference engine (Apple Silicon / Metal)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prabod Rathnayaka",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

from setuptools import find_packages, setup

__version__ = "0.1"
setup(
    name="tlal-pii",
    version=__version__,
    python_requires=">=3.9,<3.13",
    install_requires=[
        "pandas==2.2.0",
        "pyarrow==15.0.0",
        "scikit-learn==1.2.2",
        "sentencepiece==0.1.99",
        "spacy==3.7.2",
        "protobuf==3.20.3",  # required for deberta-v3 tokenizer
        "transformers==4.37.0",
        "pytorch-lightning==2.1.3",
        "tqdm==4.66.1",
    ],
    extras_require={
        "embeddings": [
            "faiss-cpu==1.7.4",
            "sentence-transformers==2.2.2",
        ],
        "lint": [
            "black==24.2.0",
            "isort==5.13.2",
            "pre-commit==3.6.1",
            "flake8==7.0.0",
            "mypy==1.8.0",
        ],
        "tests": [
            "pytest==7.4.4",
            "pytest-cov==4.1.0",
        ],
        "notebook": ["jupyterlab==4.0.11", "ipywidgets==8.1.1", "seaborn==0.12.2"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Detect AI Generated Text 2023 Kaggle Competition",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-detect-ai-generated-text-2023",
)

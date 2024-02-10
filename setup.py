from setuptools import find_packages, setup

dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "kaggle",
    "matplotlib",
    "seaborn",
    "jupyterlab",
    "pyarrow",
    "tqdm",
    "xgboost",
    "pytorch",
    "torchvision",
    "torchaudio",
    "transformers",
    "hydra-core",
    "lightning",
    "datasets",
    "black",
    "isort",
    "ipython",
    "pytest",
    "go-task",
    "nltk",
    "unzip",
    "faker",
    "imblearn",
    "spacy",
    "pydantic",
]


setup(
    name="pii_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=dependencies,
    author="Sagi Ahrac",
    author_email="sagiahrak@gmail.com",
    description="Kaggle PII detection competition package.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

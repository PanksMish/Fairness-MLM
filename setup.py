from setuptools import setup, find_packages

setup(
    name="adapt_bts",
    version="1.0.0",
    description="ADAPT-BTS: Adaptive Dual-Phase Framework for Mitigating Demographic and Dialectal Bias in Multilingual Language Models",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "accelerate>=0.23.0",
        "evaluate>=0.4.0",
        "seqeval>=1.2.2",
    ],
)

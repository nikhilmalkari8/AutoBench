from setuptools import setup, find_packages

setup(
    name="autobench",
    version="0.1",
    description="Automated Benchmarking Toolkit for ML models",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.3.1",
        "numpy==1.25.2",
    ],
)

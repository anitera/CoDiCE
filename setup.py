from setuptools import setup, find_packages

setup(
    name="trustce",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.1",
        "pandas==2.1.1",
        "scikit-learn==1.3.1",
        "scipy==1.11.3",
        "pydiffmap==0.2.0.1"
    ],
    extras_require={
        "dev": [
            "pytest>=3.4.3",
            "ipython>=7.2.0",
        ],
        "sklearn": ["scikit-learn>=0.24.1"],
        "tensorflow": ["tensorflow>=2.4.1"],
        "pytorch": ["torch>=1.8.1"],
        "gpgomea": ["gpgomea>=0.1.0"]
    },
)
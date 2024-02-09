from setuptools import setup, find_packages

setup(
    name="trustce",
    version="0.2.9",
    packages=find_packages(),
    package_data={
        "trustce": ["input_instance/*.json", "config/*.yaml", "config/*.json", "examples_notebooks_demos/*.ipynb", "models/*.pkl", "models/*.pth"]
    },
    data_files=[("", ["LICENSE", 'README.md'])],
    install_requires=[
        "numpy==1.26.1",
        "pandas==1.5.3",
        "scikit-learn==1.3.1",
        "scipy==1.11.3",
        "pydiffmap==0.2.0.1",
        "Pillow==10.0.1",
        "PyYAML==6.0.1",
        "requests==2.31.0",
        "ipykernel>=5.5.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4"
        ],
        "sklearn": ["scikit-learn>=0.24.1"],
        "tensorflow": ["tensorflow>=2.14.1"],
        "pytorch": ["torch>=1.8.1"],
        "gpgomea": ["gpgomea>=0.1.0"],
        "homeloan": [],
        "diabetes": [],
        "income": []
    }
)
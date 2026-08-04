from setuptools import setup, find_packages

setup(
    name = "gbmContacts",
    version = "1.0.0",
    description = "Antibody-Antigen interaction evaluation using gradient boosting regression methods",
    author = "Fernando Palluzzi",
    packages = find_packages(),
    python_requires = ">=3.8",
    install_requires = [
        "pandas>=3.0.2",
        "numpy>=2.4.4",
        "scipy>=1.17.1",
        "scikit-learn>=1.8.0",
        "matplotlib>=3.10.8",
        "seaborn>=0.13.2",
        "lightgbm>=4.6.0",
        "reportlab>=5.0.0"
    ],
    package_data = {
        "gbmContacts": ["data/*"],
    },
    include_package_data = True
)
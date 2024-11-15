from setuptools import setup, find_packages

setup(
    name='swfilter',
    version='0.1',
    package_dir = {"": "src"},
    packages = find_packages(where="src"),
    description='swfilter: A Python implementation of the sliced-Wasserstein-based anomaly detection method from the work of Julien Pallage et al.',
    long_description="For more information, please visit the GitHub repository: jupall/swfilter",
    install_requires=[
        'numpy>=1.26',
        'POT>=0.9.4',
        'pandas>=2.1',
        'scikit-learn>=1.3',
        'scipy>=1.11',
        'joblib>=1.3.1'],
    python_requires=">=3.8",
    url = "https://github.com/jupall/swfilter",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
        )


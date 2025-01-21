from setuptools import setup, find_packages

setup(
    name='recipes',
    version='1.0.0',
    author="Ning E",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
# setup.py
from setuptools import setup, find_packages

setup(
    name="flipkart-return-prediction",
    version="1.0.0",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
)
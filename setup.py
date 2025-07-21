from setuptools import setup, find_packages
from pathlib import Path

directory = Path(__file__).resolve().parent

setup(
    name="birdview",
    version="0.1",
    description="Like Tesla's birdview but at home",
    author="Pablo Felgueres",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0"
    ] 
)
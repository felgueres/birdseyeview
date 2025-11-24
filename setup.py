from setuptools import setup, find_packages
from pathlib import Path

directory = Path(__file__).resolve().parent

setup(
    name="birdview",
    version="0.1",
    description="BirdView",
    author="Pablo Felgueres",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "ultralytics>=8.0.0",
        "python-dotenv",
        "Pillow>=10.0.0",
        "ollama",  # For local LLaVA scene graph generation
        "openai",
        "networkx>=2.5",  # For scene graph visualization
        "matplotlib>=3.3.0",  # For scene graph plotting
        "huggingface_hub>=0.26.0",
        "modal>=1.2.4"
    ] 
)
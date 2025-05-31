"""
Setup script for Street Food Classifier package.

This allows you to install the package in development mode:
    pip install -e .

Then you can import from anywhere:
    from src.street_food_classifier import StreetFoodClassifier
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'Pillow>=8.0.0',
        'tqdm>=4.62.0',
        'pandas>=1.3.0'
    ]

setup(
    name="street_food_classifier",
    version="1.0.0",
    author="Oliver",
    author_email="your.email@example.com",
    description="Modular Deep Learning Framework for Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/street_food_classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=3.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'streetfood-train=scripts.train:main',
            'streetfood-evaluate=scripts.evaluate:main',
            'streetfood-predict=scripts.predict:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
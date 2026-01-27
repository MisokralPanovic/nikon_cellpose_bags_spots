# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def read_requirements():
    req_file = Path(__file__).parent / 'requirements.txt'
    if not req_file.exists():
        return []
    
    with open(req_file) as f:
        return [
            line.strip() 
            for line in f.readlines()
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="spot_detector",
    version="0.1.0",
    description="Automated segmentation and spot detection pipeline for microscopy images",
    author="Michal Varga",
    author_email="varga@cshl.edu",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'spot-detect=spot_detector.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'spot_detector': ['../config.yml'],
    },
)
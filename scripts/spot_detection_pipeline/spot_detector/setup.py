# setup.py (in spot_detection_pipeline folder)
from setuptools import setup, find_packages

setup(
    name="spot_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'spot-detect=spot_detector.cli:main',
        ],
    },
    python_requires='>=3.9',
)
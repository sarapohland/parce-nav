from setuptools import setup, find_packages

requirements = ['numpy',
                'matplotlib',
                'opencv-python',
                'pandas',
                'Pillow',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'seaborn',
                'shapely',
                'torch',
                'torchvision',
                'tqdm'] 

setup(
    name="navigation",
    version="1.0.0",
    description="Probabilistically Safe Navigation with Reconstruction-Based Perception Model Competency-Awareness",
    packages=find_packages(),
    install_requires=requirements
)
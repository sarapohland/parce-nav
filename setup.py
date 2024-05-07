from setuptools import setup, find_packages

requirements = ['bagpy',
                'cvxpy',
                'numpy',
                'matplotlib',
                'moviepy',
                'opencv-python',
                'pandas',
                'Pillow',
                'pyyaml',
                'rosnumpy',
                'rospkg',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'seaborn',
                'shapely',
                'tabulate',
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
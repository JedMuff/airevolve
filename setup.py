from setuptools import setup, find_packages

setup(
    name="airevolve",
    version="0.1.0",
    description="Evolutionary algorithm framework for optimizing drone morphology and control",
    author="AirEvolve Contributors",
    author_email="contact@airevolve.dev",
    url="https://github.com/yourusername/airevolve",
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "trimesh",
        "torch>=1.9.0",
        "sympy>=1.8.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "tensorboard",
        "moviepy",
    ],  # List your dependencies here
    python_requires=">=3.7",  # Specify the Python version requirement
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
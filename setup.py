from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diabetic_retinopathy_project_ml",
    version="1.0.0",
    author="Mukesh Kumar",
    author_email="mukeshloney67@gmail.com",
    description="Diabetic Retinopathy Detection & Severity Classification using InceptionV3 and TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mukeshloney67/diabetic_retinopathy_project_ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dr-train=src.train:main",
            "dr-predict=src.predict:main",
            "dr-evaluate=src.evaluate:main",
        ],
    },
)

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE", "r", encoding="utf8") as f:
    license = f.read()

setup(
    name="sicore",
    version="2.0.0",
    description="Core package for Selective Inference",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Takeuchi Lab",
    author_email="shirara1016@gmail.com",
    url="https://github.com/shirara1016/sicore",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "numpy>=1.26.4,<2.0.0",
        "mpmath>=1.3.0",
        "matplotlib>=3.9.1",
        "scipy>=1.14.0",
    ],
    python_requires=">=3.10",
    license="MIT License",
)

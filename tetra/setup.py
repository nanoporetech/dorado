from setuptools import find_packages, setup

with open("README.md") as readme:
    long_description = readme.read()

install_requirements = [
    "biopython",
    "certifi",
    "numpy",
    "pandas",
    "pod5>=0.3.33",
    "distro",
]

setup(
    name="ont-tetra",
    description="Oxford Nanopore Technologies regression testing project",
    python_requires=">=3.7",
    long_description=long_description,
    url="http://www.nanoporetech.com",
    install_requires=install_requirements,
    packages=find_packages(),
)

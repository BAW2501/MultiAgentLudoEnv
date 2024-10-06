from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Ludo",
    version="0.1.0",
    author="Belhadj Ahmed walid",
    author_email="belhadj.ahmedwalid1@gmail.com",
    description="A Ludo game environment with multiple implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BAW2501/MultiAgentLudoEnv",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "pettingzoo",
        "gymnasium",
    ],
    include_package_data=True,
    package_data={
        "Ludo": ["utils/ludo.svg"],
    },
)
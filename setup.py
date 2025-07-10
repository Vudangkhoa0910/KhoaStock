# Author: Vu Dang Khoa

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="khoastock",
    version="1.0.0",
    author="Vu Dang Khoa",
    author_email="vudangkhoa0910@gmail.com",
    description="Vietnam Stock Market Analysis & Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vudangkhoa0910/KhoaStock",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "khoastock=main:main",
        ],
    },
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrfdpd",
    version="0.0.1",
    author="Zhe Li",
    author_email="ataraxialex@gmail.com",
    description="Python package for radio frequency digital predistortion techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SEU-MSLab/pyrfdpd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pyvisa",
        "numpy",
        "scipy",
        "matplotlib",
    ],
)

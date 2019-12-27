import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

short_description = \
"""
A python package developed for symbolic creation and analysis 
of constrained multi-body systems.
"""

setuptools.setup(
    name = "uraeus",
    version = "0.0.dev1",
    author = "Khaled Ghobashy",
    author_email = "khaled.ghobashy@live.com",
    description = short_description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/khaledghobashy/uraeus",
    packages = setuptools.find_packages(exclude=("tests",)),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)
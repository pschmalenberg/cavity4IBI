#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io, os
from setuptools import find_packages, setup

# Package metadata
NAME = "cavity4ibi"
DESCRIPTION = "ECG reconstruction from acoustic Helmholtz cavity signals"
URL = "https://github.com/pschmalenberg/cavity4IBI"
AUTHOR = "Paul D. Schmalenberg et al."
REQUIRES_PYTHON = ">=3.11,<3.12"
VERSION = "1.0.0"

REQUIRED = [
    "auraloss==0.4.0",
    "dtw-python==1.5.3",
    "matplotlib==3.7.2",
    "neurokit2==0.2.5",
    "numpy==1.25.2",
    "pandas==2.0.3",
    "scipy==1.10.0",
    "torch==2.0.1",
    "torchaudio==2.0.2",
    "tqdm==4.66.0",
]

EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)

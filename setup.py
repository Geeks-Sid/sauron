# setup.py (at the root of your project)
import os
import re

import setuptools

# --- Package Metadata ---
NAME = "sauron"
DESCRIPTION = "A comprehensive deep learning framework for Whole Slide Image analysis, including feature extraction and MIL training."
URL = "https://github.com/iucomppath/sauron"  # Replace with your project's GitHub URL
AUTHOR = "Siddhesh Thakur"  # Replace with your name
AUTHOR_EMAIL = "sid.cre8er@gmail.com"  # Replace with your email
LICENSE = (
    "Apache-2.0"  # Or whatever license you are using (e.g., Apache-2.0, BSD-3-Clause)
)


# --- Version Management ---
# Load the version from sauron/__init__.py
def get_version():
    """Reads the version from sauron/__init__.py without importing the package."""
    version_file = os.path.join(os.path.dirname(__file__), NAME, "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


VERSION = get_version()

# --- Long Description ---
# Read the long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        LONG_DESCRIPTION = fh.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION  # Fallback if README.md is not found


# --- Dependencies ---
# Read requirements from requirements.txt
def get_requirements(filename="requirements.txt"):
    """Reads the list of requirements from a file."""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    with open(requirements_path, "r", encoding="utf-8") as f:
        # Filter out comments and empty lines
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


INSTALL_REQUIRES = get_requirements()

# --- Setup Configuration ---
setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    # Automatically find all packages under the 'sauron' directory
    packages=setuptools.find_packages(
        exclude=["tests*", "docs*"]
    ),  # Exclude test/doc directories
    python_requires=">=3.8",  # Specify minimum Python version
    install_requires=INSTALL_REQUIRES,
    # Include non-Python files specified in MANIFEST.in
    include_package_data=True,
    # Define command-line entry points
    entry_points={
        "console_scripts": [
            # These now point to the functions within sauron/cli.py
            "sauron-extract = sauron.cli:feature_extract_main",
            "sauron-train = sauron.cli:train_mil_main",
        ],
    },
    # Classifiers help users find your project on PyPI and understand its compatibility.
    classifiers=[
        # Development Status
        "Development Status :: 3 - Alpha",  # Or 4 - Beta, 5 - Production/Stable
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        # License
        f"License :: OSI Approved :: {LICENSE} License",
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Topic
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "whole slide imaging",
        "wsi",
        "pathology",
        "deep learning",
        "feature extraction",
        "multiple instance learning",
        "pytorch",
        "cancer research",
        "histopathology",
    ],
)

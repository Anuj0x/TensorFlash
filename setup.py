#!/usr/bin/env python3

"""
Setup script for Modern Flash Attention
=====================================

Installation:
    pip install -e .
    # or
    python setup.py develop
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import setuptools


class CMakeExtension(setuptools.Extension):
    """CMake extension for CUDA code"""

    def __init__(self, name, sourcedir=''):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command for CMake"""

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Configure with CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                            cwd=self.build_temp)

        # Build with CMake
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)


class PostInstallCommand(install):
    """Post-installation command to verify CUDA setup"""

    def run(self):
        install.run(self)

        # Try to import the extension to verify it works
        try:
            import torch
            from flash_attention import flash_attention_forward

            # Quick test
            if torch.cuda.is_available():
                print("✅ CUDA extension loaded successfully!")
                print(f"CUDA devices available: {torch.cuda.device_count()}")
            else:
                print("⚠️  CUDA not available. Extension loaded but will use CPU fallback.")

        except ImportError as e:
            print(f"❌ Failed to import extension: {e}")
            print("Make sure CUDA and PyTorch with CUDA support are installed.")


# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="modern-flash-attention",
    version="1.0.0",
    author="Modern Flash Attention Team",
    author_email="",
    description="Modern Flash Attention with backward pass, half precision, and dynamic block sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/modern-flash-attention",

    packages=find_packages(where="python"),
    package_dir={"": "python"},

    ext_modules=[CMakeExtension('minimal_attn')],
    cmdclass={
        'build_ext': CMakeBuild,
        'install': PostInstallCommand,
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    python_requires=">=3.8",
    install_requires=read_requirements(),

    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },

    keywords="attention transformer cuda pytorch deep-learning nlp",

    project_urls={
        "Bug Reports": "https://github.com/your-repo/modern-flash-attention/issues",
        "Source": "https://github.com/your-repo/modern-flash-attention",
        "Documentation": "https://modern-flash-attention.readthedocs.io/",
    },

    zip_safe=False,
)

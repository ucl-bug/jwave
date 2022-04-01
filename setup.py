"""Python setup.py for jwave package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
  """Read the contents of a text file safely.
  >>> read("jwave", "VERSION")
  '0.1.0'
  >>> read("README.md")
  ...
  """

  content = ""
  with io.open(
    os.path.join(os.path.dirname(__file__), *paths),
    encoding=kwargs.get("encoding", "utf8"),
  ) as open_file:
    content = open_file.read().strip()
  return content


def read_requirements(path):
  return [
    line.strip()
    for line in read(path).split("\n")
    if not line.startswith(('"', "#", "-", "git+"))
  ]

setup(
    name="jwave",
    version=read("jwave", "VERSION"),
    description="Fast, differentiable acoustic simulations in JAX",
    url="https://github.com/ucl-bug/jwave",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["tests", ".github", "docs"]),
    python_requires=">=3.7",
    install_requires=read_requirements(".requirements/requirements.txt"),
    extras_require={
      "test": read_requirements(".requirements/requirements-test.txt"),
    },
    license="GNU Lesser General Public License (LGPL)",
    keywords=[
      "jax","acoustics","simulation", "ultrasound",
      "differentiable-programming"
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

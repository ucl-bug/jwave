from setuptools import setup, find_packages

# Get version
_dct = {}
with open("jwave/version.py") as f:
    exec(f.read(), _dct)
__version__ = _dct["__version__"]

setup(
    name="j-Wave",
    version=__version__,
    description="Fast differentiable pseudospectral acoustic simulations",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["docs"]),
    package_data={"jwave": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=open("requirements.txt", "r").readlines(),
    extras_require={
        "dev": open("dev_requirements.txt", "r").readlines(),
    },
    url="https://bug.medphys.ucl.ac.uk/",
    license="GNU Lesser General Public License (LGPL)",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

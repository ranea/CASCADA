import os
import sys
import pathlib

from setuptools import find_packages, setup
from setuptools.command.install import install as _install

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'cascada'
AUTHOR = 'Adrián Ranea'
AUTHOR_EMAIL = 'ranea@protonmail.com'
URL = 'https://github.com/ranea/CASCADA'

LICENSE = 'MIT'
DESCRIPTION = 'CASCADA (Characteristic Automated Search of Cryptographic Algorithms for Distinguishing Attacks) is a Python 3 library to evaluate the security of cryptographic primitives, specially block ciphers, against distinguishing attacks with bit-vector SMT solvers.'
LONG_DESCRIPTION = (HERE / "README.rst").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'cython',
    'sympy',
    'bidict',
    'cffi',
    'wurlitzer',
    'pySMT'
    ]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
)

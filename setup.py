#!/usr/bin/env python

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="dddm_vc",
    version="0.0.1",
    description="DDDM-VC pip installable",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=find_packages(),
    install_requires=required
)

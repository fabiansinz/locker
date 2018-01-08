#!/usr/bin/env python
from glob import glob
from os.path import basename

from setuptools import setup, find_packages
from os import path, listdir

here = path.abspath(path.dirname(__file__))

long_description = "Analysis code for efish locker. "


setup(
    name='locker',
    version='0.1.0.dev1',
    description="Analysis code for efish locker",
    long_description=long_description,
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License",
    url='https://github.com/fabiansinz/locker',
    keywords='Analysis code for publication',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy','matplotlib','seaborn','scipy','statsmodels==0.8.0','sympy==0.7.6.1','Pint==0.6',
                      'PyYAML==3.10','nose==1.3.1'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License',
        'Topic :: Analysis :: Reproducibility',
    ],
    scripts=['scripts/{0}'.format(basename(file)) for file in glob('scripts/*.py')]

)

##############################################################################
# fbmp2 is a python3 package that implements an improved version of
# Fast Bayesian Matching Pursuit, a variable selection algorithm for
# linear regression
#
# Copyright (C) 2020  Totient, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License v3.0
# along with this program.
# If not, see <https://www.gnu.org/licenses/gpl-3.0.en.html>.
#
# Developer:
# Peter Komar (peter.komar@totient.bio)
##############################################################################

__author__ = 'Peter Komar "\
             "<peter.komar@totient.bio>'
__copyright__ = '2020 Totient, Inc'
__version__ = '1.0.0'

import io
from datetime import datetime
from setuptools import setup, find_packages

setup(
    name='fbmp2',
    version=__version__,
    description='Fast Bayesian Matching Pursuit with improvements.',
    long_description=io.open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX',
    ],
    author='Peter Komar',
    author_email='peter.komar@totient.bio',
    maintainer='Peter Komar',
    maintainer_email='peter.komar.hu@gmail.com',
    url='https://github.com/totient-bio/fbmp2',
    license='Copyright (c) {} Totient, Inc.'.format(
        datetime.now().year
    ),
    packages=find_packages(),
    install_requires=io.open('requirements.txt').read().splitlines(),
    include_package_data=True,
    scripts=["command-line-tool/fbmp2-feature-selection.py"],
    python_requires='>=3.6'
)
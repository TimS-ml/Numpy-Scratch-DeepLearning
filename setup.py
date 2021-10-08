from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.5'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')
]

setup(
    name='scratchDL',
    version=__version__,
    description=
    'Numpy implementations of Deep Learning from scratch.',
    url='https://github.com/TimS-ml/Scratch-DL',
    download_url=
    'https://github.com/TimS-ml/Scratch-DL/tarball/master',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Tim Shen',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10'],
    dependency_links=dependency_links,
    author_email='jshen44@fordham.edu')

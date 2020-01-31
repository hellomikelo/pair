from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['scikit-learn>=0.20.4', 'pandas>=0.24.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)

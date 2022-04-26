from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Wavelet transforms and associated functions'
LONG_DESCRIPTION = 'Wavelet transforms and associated functions'

# Setting up
setup(
        name='wavelets',
        version=VERSION,
        author='Matyas Molnar',
        author_email='mdm49@cam.ac.uk',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'wavelets', 'multiresolution']
)

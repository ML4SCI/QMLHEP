from setuptools import setup

CLASSIFIERS = '''\
License :: OSI Approved
Programming Language :: Python :: 3.9
Topic :: Software Development
'''

DISTNAME = 'hep_VQAE'
AUTHOR = 'Tom Magorsch'
AUTHOR_EMAIL = 'tom.magorsch@tu-dortmund.de'
DESCRIPTION = 'Quantum (Variational) Autoencoder for hep data analysis'
LICENSE = 'MIT'
README = 'Quantum (Variational) Autoencoder for hep data analysis'

VERSION = '0.1.0'
ISRELEASED = False

PYTHON_MIN_VERSION = '3.9'
PYTHON_REQUIRES = f'>={PYTHON_MIN_VERSION}'

PACKAGES = [
    'hep_VQAE',
]

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE
)

if __name__ == '__main__':
    setup(**metadata)

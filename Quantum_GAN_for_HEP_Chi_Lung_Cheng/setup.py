import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE="quple/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))        

setuptools.setup(
    name="quple", # Replace with your own username
    version=verstr,
    author="Alkaid Cheng",
    author_email="chi.lung.cheng@cern.ch",
    description="A framework for quantum machine learning in high energy physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cern.ch/clcheng/quple",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[   
          'cirq',
          'numpy',
          'imageio',
          'tensorflow==2.4.1',
          'tensorflow_quantum'
      ],
    python_requires='>=3.5',
)

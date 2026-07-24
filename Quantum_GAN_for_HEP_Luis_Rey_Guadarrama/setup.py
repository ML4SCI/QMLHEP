from setuptools import setup, find_packages

setup(
    name="python_package",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
)
from setuptools import setup, find_packages

setup(
    name="qml_contrastive",
    version="0.1",
    package_dir={'': 'src'},
    packages= [
        'qml_contrastive',
    ],
    install_requires=[
        "pennylane",
    ],

)

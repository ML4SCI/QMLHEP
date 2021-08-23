#!/bin/bash
rm -r dist
rm -r build
rm -r quple.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*

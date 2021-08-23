# MIT License
#
# Copyright (c) 2021 Eraraya Ricardo Muten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from distutils.core import setup

setup(
    name='qcnn_drc',
    version='1.0.0',
    author='Eraraya Ricardo Muten',
    author_email='eraraya-ricardo@qlab.itb.ac.id',
    description="Tensorflow Quantum implementation of Quantum Convolutional Neural Networks",
    long_description="This package is a TensorFlow Quantum implementation of quantum convolution and classifier with Data Re-uploading ansatz. Both are wrapped as Keras layers that can easily be integrated into other Keras layers (classical and/or quantum), acting as building blocks for Quantum Convolutional Neural Networks (both hybrid and fully quantum). The model can be trained using Keras API.",
    packages=['qcnn_drc'],
    url='https://github.com/eraraya-ricardo/qcnn-hep'
)

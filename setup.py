import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("AlphaGo.go", ["AlphaGo/go.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go_data", ["AlphaGo/go_data.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.preprocessing.preprocessing", ["AlphaGo/preprocessing/preprocessing.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
]

setup(name="RocAlphaGo", ext_modules=cythonize(extensions))

"""
   install all necessary dependencies using:
   pip install -r requirements.txt

   run setup with command:
   python setup.py build_ext --inplace

   be aware cython uses a depricaped version of numpy this results in a lot of warnings

   you can run all unittests to verify everything works as it should:
   python -m unittest discover
"""

import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("AlphaGo.go.constants", ["AlphaGo/go/constants.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go.ladders", ["AlphaGo/go/ladders.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go.game_state", ["AlphaGo/go/game_state.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go.group_logic", ["AlphaGo/go/group_logic.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go.coordinates", ["AlphaGo/go/coordinates.pyx"],
              include_dirs=[numpy.get_include()], language="c++"),
    Extension("AlphaGo.go.zobrist", ["AlphaGo/go/zobrist.pyx"],
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

import numpy
import os
from distutils.core import setup
from Cython.Build import cythonize

setup(

    name='RocAlphaGo',
    # list with files to be cythonized
    ext_modules=cythonize(["AlphaGo/go.pyx", "AlphaGo/go_data.pyx",
                           "AlphaGo/preprocessing/preprocessing.pyx",
                           "AlphaGo/preprocessing/preprocessing_rollout.pyx"]),
    # include numpy
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')]
)

"""
   install all necessary dependencies using:
   pip install -r requirements.txt

   run setup with command:
   python setup.py build_ext --inplace

   be aware cython uses a depricaped version of numpy this results in a lot of warnings

   you can run all unittests to verify everything works as it should:
   python -m unittest discover
"""

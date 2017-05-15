import numpy
import os
from distutils.core import setup
from Cython.Build import cythonize

setup(
    
    name = 'RocAlphaGo',
    # list with files to be cythonized
    ext_modules = cythonize( [ "AlphaGo/go.pyx", "AlphaGo/go_root.pyx", "AlphaGo/go_data.pyx", "AlphaGo/preprocessing/preprocessing.pyx" ] ),
    # include numpy
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')]
)

# run setup with command
# python setup.py build_ext --inplace

# be aware cython uses a depricaped version of numpy this results in a lot of warnings
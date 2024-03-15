import os
from distutils.core import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_ext import build_ext


here = os.path.dirname(os.path.abspath(__file__))


def find_pyx(path=None):
    path = path or here
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files


def find_cython_extensions(path=None):
    extensions = []
    for item in find_pyx(path):
        relpath = os.path.relpath(os.path.abspath(item), start=here)
        rpath, _ = os.path.splitext(relpath)
        extname = '.'.join(rpath.split(os.path.sep))
        extensions.append(Extension(
            name=extname,
            sources=[relpath],  # build file in current dir: core/mcts/ctree/build
            # sources=[item],  # build file in root dir: /build
            language="c++",
            extra_compile_args=['-Wall', '-Wextra', "-fopenmp", "-O2"],
        ))
    return extensions


# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()


setup(
    cmdclass={'build_ext': BuildExt},
    ext_modules=cythonize(
        find_cython_extensions(),
        language_level=3,
    ),
)

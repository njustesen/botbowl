from setuptools import setup, find_packages
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(name='ffai',
      version="0.3.2",
      include_package_data=True,
      install_requires=[
          'numpy',
          'untangle',
          'Flask',
          'gym',
          'Jinja2',
          'python-interface',
          'stopit',
          'requests',
          'pytest',
          'matplotlib'
      ],
      packages=find_packages(),
      ext_modules=cythonize(["ffai/ai/fast_pathing.pyx"], annotate=True),
      zip_safe=False
)


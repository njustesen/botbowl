from setuptools import setup, find_packages

try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    cython_exists = True
except ImportError:
    cython_exists = False

files_to_compile = ["ffai/ai/fast_pathing.pyx"]

install_requires_packages = [
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
]

kwargs = {
    'name': 'ffai',
    'version': '0.3.2',
    'include_package_data': True,
    'install_requires': install_requires_packages,
    'packages': find_packages(),
    'zip_safe': False
}

if cython_exists:
    kwargs['ext_modules'] = cythonize(files_to_compile, annotate=True)

setup(**kwargs)


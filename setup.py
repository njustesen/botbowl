from setuptools import setup, find_packages
import os, shutil, platform
import sysconfig

try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    compile_available = 'CXX' in sysconfig.get_config_vars() # Is c++ compiler available?
except ImportError:
    compile_available = False
    
files_to_compile = ["botbowl/core/pathfinding/cython_pathfinding.pyx"]

install_requires_packages = [
          'numpy',
          'untangle',
          'Flask',
          'gym',
          'Jinja2',
          'python-interface',
          'stopit',
          'requests',
          'Cython >= 3.0a7',
          'pytest',
          'matplotlib'
]

kwargs = {
    'name': 'botbowl',
    'version': '0.4.0',
    'include_package_data': True,
    'install_requires': install_requires_packages,
    'packages': find_packages(),
    'zip_safe': False
}

if compile_available:
    kwargs['ext_modules'] = cythonize(files_to_compile, annotate=True)

setup(**kwargs)

if compile_available:
    # Grab all compiled modules and copy into source folders
    compiled_file_type = ".pyd" if platform.system() == "Windows" else ".so"
    copied_files = 0
    for root, dirs, files in os.walk('./build/'):
        for file in files:
            if file.endswith(compiled_file_type):
                from_file = f"{root}/{file}"
                to_file = "./botbowl/" + root.split('/botbowl/')[1] + "/" + str(file)
                print(f"copying '{from_file}' -> '{to_file}'")
                shutil.copyfile(from_file, to_file)
                copied_files += 1

    #assert copied_files == len(files_to_compile), f"Wrong number if files copied. " \
    #                                              f"Supposed to copy {len(files_to_compile)} files " \
    #                                              f"but {copied_files} was copied. Probably a bug!"
    print("\nYou've built botbowl with cython.")
else:
    print("You've built botbowl without cython compilation. Check docs/installation.md for details.")

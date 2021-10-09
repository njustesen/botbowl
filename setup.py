from setuptools import setup, find_packages
import os, shutil

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

if cython_exists:
    # Grab all '.so'-files and copy into source folders
    for root, dirs, files in os.walk('./build/'):
        for file in files:
            if file.endswith('.so'):
                so_file = f"{root}/{file}"
                to_file = "./ffai/" + root.split('/ffai/')[1] + "/" + str(file)
                print(f"copying '{so_file}' -> '{to_file}'")
                shutil.copyfile(so_file, to_file)

    print("\nYou've built FFAI with cython.")

else:
    print("You've built FFAI without cython compilation. Check docs/installation.md for details.")


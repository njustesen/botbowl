from setuptools import setup, find_packages
import os, shutil, platform
import sysconfig

can_compile = 'CXX' in sysconfig.get_config_vars()
if can_compile:
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True

files_to_compile = ["ffai/ai/fast_pathing.pyx"]
compiled_file_type = ".pyd" if platform.system() == "Windows" else ".so"

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
    'name': 'ffai',
    'version': '0.3.2',
    'include_package_data': True,
    'install_requires': install_requires_packages,
    'packages': find_packages(),
    'zip_safe': False
}


if can_compile:
    kwargs['ext_modules'] = cythonize(files_to_compile, annotate=True)

setup(**kwargs)


if can_compile:
    # Grab all '.so'-files and copy into source folders
    copied_files = 0
    for root, dirs, files in os.walk('./build/'):
        for file in files:
            if file.endswith(compiled_file_type):
                so_file = f"{root}/{file}"
                to_file = "./ffai/" + root.split('/ffai/')[1] + "/" + str(file)
                print(f"copying '{so_file}' -> '{to_file}'")
                shutil.copyfile(so_file, to_file)
                copied_files += 1

    assert copied_files == len(files_to_compile), f"Compiled with strange result, didn't copy corrent amount of files!"
    print("\nYou've built FFAI with cython.")

else:
    print("You've built FFAI without cython compilation. Check docs/installation.md for details.")


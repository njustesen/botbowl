from setuptools import setup, find_packages

setup(name='ffai',
      version="0.0.6",
      include_package_data=True,
      install_requires=[
          'numpy',
          'untangle',
          'Flask',
          'gym',
          'Jinja2',
          'python-interface'
      ],
      packages=find_packages()
)

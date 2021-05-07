from setuptools import setup, find_packages

setup(name='ffai',
      version="0.2.1",
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
          'torch',
          'torchvision',
          'matplotlib'
      ],
      packages=find_packages()
)

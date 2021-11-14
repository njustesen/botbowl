# Installation
botbowl is a pip-installable python package. We recommend installing with [Anaconda](https://docs.anaconda.com/anaconda/install/) but it is not a requirement. 

## Python version
botbowl currently works with python 3.6 - 3.9  
Verify your python version with 
```
python --version
```
If you just installed Anaconda, create and activate a new environment. 
```
conda create --name botbowl python=3.7
conda activate botbowl
```

## Installation with pip
```
pip install git+https://github.com/njustesen/botbowl
```
Or, use ```pip3``` if your pip points to a python 2 installation.

## Installation with git

Alternatively, if you want to run our examples or develop on botbowl, you can clone the repository and install it locally.
```
git clone https://github.com/njustesen/botbowl
cd botbowl
python setup.py build
pip install -e .
```
To test the installation, run the following:
```
python -c "import botbowl"
```
This should not produce an import error.

If you cloned the repository and installed locally, you can start the web server:
```
python examples/server_example.py 
```
and then go the [http://127.0.0.1:1234/](http://127.0.0.1:1234/) to play a game.

If you ran into issues, please seek help at our [Discord server](https://discord.gg/MTXMuae).

## Compiling with Cython
"[Cython](https://github.com/cython/cython) is a language that makes writing C extensions for Python as easy as Python itself."

Some of the botbowl modules are prepared for compilation with Cython. Compiling them will make the framework run faster. You need to have a C++ compiler available. Check the [Cython documentation](https://cython.readthedocs.io/en/latest/) for more information. 

Installation through pip will automatically build with Cython if possible. If you cloned the repository with git, you need to manually call the build script by standing in the root of the repo and running:
```
python setup.py build 
```
You get a message confirming that botbowl was built with Cython. To test simply try to import one of the compiled modules with `import botbowl.core.pathfinding.cython_pathfinding`. You're good to go! 

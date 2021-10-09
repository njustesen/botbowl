# Installation
FFAI is a pip-installable python package. We recommend installing with [Anaconda](https://docs.anaconda.com/anaconda/install/) but it is not a requirement. 

## Python version
FFAI currently works with python 3.6 and 3.7.
Verify your python version with 
```
python --version
```
If you just installed Anaconda, create and activate a new environment. 
```
conda create --name ffai python=3.7
conda activate ffai
```

## Installation with pip
```
pip install git+https://github.com/njustesen/ffai
```
Or, use ```pip3``` if your pip points to a python 2 installation.

Alternatively, if you want to run our examples or develop on FFAI, you can clone the repository and install it locally.
```
git clone https://github.com/njustesen/ffai
cd ffai
pip install -e .
```
To test the installation, run the following:
```
python -c "import ffai"
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

Some of the FFAI modules are prepared for compilation with Cython. If you compile these modules the framework will run faster. Cython installs as a dependency to FFAI but you need to have a C++ available. Check the [Cython documentation](https://cython.readthedocs.io/en/latest/) for more information. 

Build FFAI by running this in the root of the repository: 
```
python setup.py build 
```
You should get a message confirming that FFAI was built with Cython. To test simply try to import one of the compiled modules with `import ffai.ai.fast_pathing`. 

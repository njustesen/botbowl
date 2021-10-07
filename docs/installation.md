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

Some modules are prepared for compilation with Cython. If you compile these modules they will run faster. 
This is interesting if you use reinforcement learning or other algorithms that need to play a lot of games between bots. 
To set up the compilation make sure Cython>=3.0.0 is installed. This will install the cython git repo with pip:  
```
pip3 install git+https://github.com/cython/cython.git
```

Then make sure you have a C++ compiler installed. Check the Cython documentation if you're unsure. 

With Cython and the C++ compiler installed we are ready to go. Build FFAI by running this in the root of the repository: 
```
python setup.py build 
```
The setup script worked you should get a message confirming that FFAI was built with Cython. 
You may need to copy the compiled module from the `build/` directory to the source library to get it working if 
you are not running `python setup.py install` afterwords. Here's how that can look on Linux for the fast pathfinding module. 
```
cp build/lib.linux-x86_64-3.8/ffai/ai/fast_pathing.cpython-38-x86_64-linux-gnu.so ffai/ai/
```

To test that everything works. Simply import the module in python with `import ffai.ai.fast_pathing`. 
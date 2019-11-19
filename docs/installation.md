# Installation
FFAI is a pip-installable python package. We recommend installing with Anaconda but it is not a requirement.

## Python version
FFAI is currently works with python 3.6 and 3.7.
Verify your python version with 
```
python --version
```

## Installation with pip
```
pip install git+https://github.com/njustesen/ffai
```
Or, use ```pip3``` if this installs it for python 2.

Alternatively, if you want to run the examples or develop on FFAI, you can clone the repository and install it locally.
```
git clone https://github.com/njustesen/ffai
cd ffai
pip install -e .
```

## Installation with Anaconda
Make sure [Anaconda](https://docs.anaconda.com/anaconda/install/) is installed.
Then setup an environment and install FFAI:
```
conda create --name ffai-test python=3.7
conda activate ffai-test
git clone https://github.com/njustesen/ffai.git
cd ffai/
pip install -e .
```
To test the installation start the web server:
```
python examples/server_example.py 
```
and go the http://127.0.0.1:5000/ to play a game.

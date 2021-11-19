# Development Guide
We'd love your help with testing, bug fixing and developing framework. We gather all improvement ideas and bugs in the [issue tracker](https://github.com/njustesen/botbowl/issues). Issues labeled ["good first issue"](https://github.com/njustesen/botbowl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are small in scope and a good way to get familiar with the framework. We also need help adding features and tests, check [features.md](features.md) for a complete list. 

Please join our Discord channel to discuss the development of botbowl [botbowl Discord Server](https://discord.gg/MTXMuae).

## Install for development
You can install botbowl with pip using the -e option inorder to test your modifcations:
```
git clone https://github.com/njustesen/botbowl
cd botbowl
python setup.py build
pip install -e .
```
Run the above command from the root of the repository. The third line `python setup.py build` will compile with cython, see [installation.md](docs/installation.md) for more info. 

## Run tests
Install pytest and run the unit and integration tests in [tests/](../tests) by running:
```
pytest
```
from the root of the repository.

Before making a pull request, please make sure that all tests pass. You should also consider if the changes you have made requires a new test.


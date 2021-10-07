# Development Guide
We'd love your help with testing, bug fixing and developing framework. We gather all improvement ideas and bugs in the [issue tracker](https://github.com/njustesen/ffai/issues). Issues labeled ["good first issue"](https://github.com/mrbermell/ffai/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) are small in scope and a good way to get familiar with the framework. We also need help [writing tests for the framework](https://github.com/mrbermell/ffai/issues/34). 

Please join our Discord channel to discuss the development of FFAI [FFAI Discord Server](https://discord.gg/MTXMuae).

## Install for development
You can install FFAI with pip using the -e option inorder to test your modifcations:
```
pip install -e .
```
Run the above command from the root of the repository.

## Run tests
Install pytest and run the unit and integration tests in [tests/](../tests) by running:
```
pytest
```
from the root of the repository.

Before making a pull request, please make sure that all tests pass. You should also consider if the changes you have made requires a new test.


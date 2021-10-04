# Development Guide
We'd love your help with testing, bug fixing and developing framework. Good places to start are issues labeled ["good first issue"](https://github.com/mrbermell/ffai/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or [writing tests for the framework](https://github.com/mrbermell/ffai/issues/34). All issue labels are found [here](https://github.com/mrbermell/ffai/labels). 

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


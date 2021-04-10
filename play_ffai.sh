#!/bin/bash

source ~/venvs/ffai_env/bin/activate
pip install -e .
python ./examples/server_example.py
deactivate


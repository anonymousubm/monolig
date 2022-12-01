#!/bin/bash

python setup.py clean
rm -rf build
rm -rf pcdet.egg-info

python setup.py develop
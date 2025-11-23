#!/bin/bash

mkdir code/build
cd code/build
cmake ..
make
cd ..
rm -r build
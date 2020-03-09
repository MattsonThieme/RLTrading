#!/bin/bash
yes 'y' | conda create -n rltest -c intel python=3
source activate rltest
yes 'y' | conda install numpy
yes 'y' | conda install pytorch torchvision -c pytorch
source deactivate

#!/bin/bash
yes 'y' | conda create -n rltrade -c intel python=3
source activate rltrade
yes 'y' | conda install numpy
yes 'y' | conda install pytorch torchvision -c pytorch
pip install ccxt
source deactivate

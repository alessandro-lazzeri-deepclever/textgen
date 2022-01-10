from pathlib import Path
import re
import json
from sklearn.model_selection import train_test_split
import random

datasetPath = Path('../dataset/ovidio/raw')
destPath = Path('../dataset/ovidio/processed')

dataset = []

mythPaths = [x for x in datasetPath.iterdir()]
train_ratio = .9
random.shuffle(mythPaths)
split_id = int(len(mythPaths) * train_ratio)

def build(mythPaths, destPath):
    for mythPath in mythPaths:
        myth = ''
        with open(mythPath, 'r', encoding='utf8') as oof:
            for l in oof.readlines():
                if l != '\n':
                    myth += str(l)
        myth += '\n'

    with open(destPath,'w', encoding='utf8') as f:
        f.write(myth)

build(mythPaths[:split_id], destPath.joinpath('train.txt'))
build(mythPaths[split_id:], destPath.joinpath('test.txt'))

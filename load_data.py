# Python 
import os

# Numpy and Pandas
import pandas as pd
import numpy as np

ROOT_DIR = os.getcwd()
DATA ='data_challenge'
DATA_DIR = os.path.join(ROOT_DIR, DATA)
TRAIN="train"
TEST="test"
SOURCE="source"
TARGET="prix"

train = pd.read_csv(os.path.join(DATA_DIR, 'boites_medicaments_train.csv'),
                    encoding='utf-8',
                    sep=';')

test = pd.read_csv(os.path.join(DATA_DIR, 'boites_medicaments_test.csv'),
                   encoding='utf-8', 
                   sep=';')

train[SOURCE] = TRAIN
test[SOURCE] = TEST
BIG = pd.concat([train, test], ignore_index=True)

print "TRAIN", train.shape
print "TEST ", test.shape
print "BIG ", BIG.shape
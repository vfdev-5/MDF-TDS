# -*- coding: UTF-8 -*-  
# Python 
import os
import re

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

### Fix 'substances' data : 
#
# Problem 1 : In all substance entries number of brackets should be pair
# For example, row 1301 (test) or 4182 (train) 'VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE B/PHUKET/3073/2013 , VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE A/CALIFORNIA/7/2009 (H1N1) PDM09 - SOUCHE ANALOGUE (A/CHRISTCHURCH/16/2010, NIB-74XP , VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE A/SWITZERLAND/9715293/2013 (H3N2) - SOUCHE ANALOGUE (A/SWITZERLAND/9715293/2013, NIB-88)' has impair number 
#
#

# Show line where number of brackets is not pair
substances = BIG['substances']

info = []
for index, s in enumerate(substances):
    nb_left_brackets = len(re.findall('\(', s))
    nb_right_brackets = len(re.findall('\)', s))
    if nb_left_brackets != nb_right_brackets:
        info.append([index, nb_left_brackets, nb_right_brackets])
        #print index, s, nb_left_brackets, nb_right_brackets

if len(info) > 0:
    print "Fix problem 1 : non matching brackets"
    for index, nb_left, nb_right in info:    
        print substances[index], nb_left, nb_right
        # Manual FIX : Replace 'NIB-74XP ,' -> 'NIB-74XP) ,'
        BIG.loc[index, 'substances'] = substances[index].replace(u'NIB-74XP ,', u'NIB-74XP) ,')
        
#
# Problem 2 : Entry of type 'VIRUS DE xxxx, {INACTIVÉ|xxx}, SOUCHE xxx' should be 'VIRUS DE xxxx {INACTIVÉ|xxx} SOUCHE xxx'
#


#
# Problem 3 : 'ANTIGÈNE DE BORDETELLA PERTUSSIS : PERTACTINE, ANATOXINE TÉTANIQUE' => ANTIGÈNE DE BORDETELLA PERTUSSIS PERTACTINE ANATOXINE TÉTANIQUE
#


#
# Problem 4 : 'xxxx : xxxx, xxxx, xxxx , xxxx' -> 'xxxx : xxxx, xxxx, xxxx , xxxx'
# For example : ANTIGÈNE COQUELUCHEUX : ANATOXINE , ANATOXINE DIPHTÉRIQUE
#

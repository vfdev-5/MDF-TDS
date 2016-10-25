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
# Problem 0 : Replace 'É', 'È', 'Ï', 'Ê', 'À'
#
#print "Fix problem 0"
substances = BIG['substances']
BIG['substances'] = substances.str.replace(u'É', 'E')
BIG['substances'] = substances.str.replace(u'Ê', 'E')
BIG['substances'] = substances.str.replace(u'È', 'E')
BIG['substances'] = substances.str.replace(u'Ï', 'I')
BIG['substances'] = substances.str.replace(u'À', 'A')



substances = BIG['substances']
for index, s in enumerate(substances):
           
    #
    # Problem 1 : In all substance entries number of brackets should be pair
    # For example, row 1301 (test) or 4182 (train) 'VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE B/PHUKET/3073/2013 , VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE A/CALIFORNIA/7/2009 (H1N1) PDM09 - SOUCHE ANALOGUE (A/CHRISTCHURCH/16/2010, NIB-74XP , VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE A/SWITZERLAND/9715293/2013 (H3N2) - SOUCHE ANALOGUE (A/SWITZERLAND/9715293/2013, NIB-88)' has impair number 
    #
    #
    
    nb_left_brackets = len(re.findall('\(', s))
    nb_right_brackets = len(re.findall('\)', s))
    if nb_left_brackets != nb_right_brackets:
        #print "Fix problem 1 : non matching brackets"        
        # Manual FIX : Replace 'NIB-74XP ,' -> 'NIB-74XP) ,'
        s = s.replace(u'NIB-74XP ,', u'NIB-74XP) ,')        
        
    #
    # Problem 2 : Entry of type 'VIRUS [xxxx, ...,] xxxx, SOUCHE xxx' should be 'VIRUS DE xxxx {INACTIVÉ|xxx} SOUCHE xxx'
    # e.g. : VIRUS DE LA GRIPPE FRAGMENTÉ, INACTIVÉ, SOUCHE B/PHUKET/3073/2013 
    #    
    groups = re.findall(r'VIRUS [\w\s,]+, SOUCHE', s)
    if len(groups) > 0:        
        for g in groups:
            repl = g.replace(',', ' ')
            s = s.replace(g, repl)
        
        
    groups = re.findall(r'(, VIVANT|, ATTENUE)', s)
    if len(groups) > 0:        
        for g in groups:
            repl = g.replace(',', ' ')
            s = s.replace(g, repl)

    #        
    # Problem 3 : Remove commas in text between brackets : 
    #    
    groups = re.findall(r'\(.*?\)', s)    
    if len(groups) > 0:
        for g in groups:
            repl = g.replace(',', ' ')
            s = s.replace(g, repl)            
                                          
    BIG.loc[index, 'substances'] = s

    
    
# Remove : (,),D',DE,LA,DU    
substances = BIG['substances']
BIG['substances'] = substances.str.replace('(', '')
BIG['substances'] = substances.str.replace(')', '')
BIG['substances'] = substances.str.replace(' D\'', '')
BIG['substances'] = substances.str.replace(' DE', '')
BIG['substances'] = substances.str.replace(' LA', '')
BIG['substances'] = substances.str.replace(' DU', '')

# custom replace
BIG['substances'] = substances.str.replace('ISPAGHUL GRAINE, TEGUMENT', 'ISPAGHUL GRAINE TEGUMENT')
    

    
    
    
# FOR DEBUG
#for index, s in enumerate(BIG['substances'].sort_values().unique()):
#    print index, s


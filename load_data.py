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

### Fix 'libelle' data : 
BIG['libelle'] = BIG['libelle'].str.replace(u'é', 'e')
BIG['libelle'] = BIG['libelle'].str.replace(u'ê', 'e')
BIG['libelle'] = BIG['libelle'].str.replace(u'è', 'e')
BIG['libelle'] = BIG['libelle'].str.replace(u'ï', 'i')
BIG['libelle'] = BIG['libelle'].str.replace(u'î', 'i')
BIG['libelle'] = BIG['libelle'].str.replace(u'à', 'a')
BIG['libelle'] = BIG['libelle'].str.replace(u'â', 'a')
BIG['libelle'] = BIG['libelle'].str.replace(u'\x92', ' ')

# Remove pluriel '(s)', e.g. comprime(s) -> comprime
BIG['libelle'] = BIG['libelle'].str.replace('\(s\)', '')

# Remove pluriel for words : e.g. comprimes -> comprime
BIG['libelle'] = BIG['libelle'].str.replace('aiguiilles', 'aiguille')
BIG['libelle'] = BIG['libelle'].str.replace('aiguilles', 'aiguille')
BIG['libelle'] = BIG['libelle'].str.replace('comprimes', 'comprime')
BIG['libelle'] = BIG['libelle'].str.replace('dispositifs', 'dispositif')
BIG['libelle'] = BIG['libelle'].str.replace('doses', 'dose')
BIG['libelle'] = BIG['libelle'].str.replace('mlavec', 'ml avec')
BIG['libelle'] = BIG['libelle'].str.replace('gelules', 'gelule')
BIG['libelle'] = BIG['libelle'].str.replace('plaquettes', 'plaquette')
BIG['libelle'] = BIG['libelle'].str.replace(r'^aquette', 'plaquette')
BIG['libelle'] = BIG['libelle'].str.replace('stylos', 'stylo')
BIG['libelle'] = BIG['libelle'].str.replace('sacs', 'sac')
BIG['libelle'] = BIG['libelle'].str.replace('spatules', 'spatule')
BIG['libelle'] = BIG['libelle'].str.replace('seringues', 'seringue')
BIG['libelle'] = BIG['libelle'].str.replace('pipettes', 'pipette')
BIG['libelle'] = BIG['libelle'].str.replace('poches', 'poche')
BIG['libelle'] = BIG['libelle'].str.replace('pansements', 'pansement')
BIG['libelle'] = BIG['libelle'].str.replace('tubes', 'tube')
BIG['libelle'] = BIG['libelle'].str.replace('cuilleres-mesure', 'cuillere')
BIG['libelle'] = BIG['libelle'].str.replace('cuillere-mesure', 'cuillere')
BIG['libelle'] = BIG['libelle'].str.replace(r'a \d compartiments', 'a compartiments')
BIG['libelle'] = BIG['libelle'].str.replace('microgrammes', 'mg')
BIG['libelle'] = BIG['libelle'].str.replace('mL', 'ml') 
BIG['libelle'] = BIG['libelle'].str.replace(' un ', ' 1 ') 
BIG['libelle'] = BIG['libelle'].str.replace(' deux ', ' 2 ') 
BIG['libelle'] = BIG['libelle'].str.replace('pour chacun', '') 
BIG['libelle'] = BIG['libelle'].str.replace('avec ou sans', '') 
BIG['libelle'] = BIG['libelle'].str.replace('systeme de recuperation', 'systeme_de_recuperation') 
BIG['libelle'] = BIG['libelle'].str.replace('systeme de securite', 'systeme_de_securite') 
BIG['libelle'] = BIG['libelle'].str.replace('fermeture de securite enfant', 'fermeture')
BIG['libelle'] = BIG['libelle'].str.replace('materiel de perfusion', 'materiel_de_perfusion') 
BIG['libelle'] = BIG['libelle'].str.replace('Trousse de ', '') 
BIG['libelle'] = BIG['libelle'].str.replace('necessaire d administration', '1 seringue et 1 catheter') 
BIG['libelle'] = BIG['libelle'].str.replace('site d\'addition', 'site_d_addition') 
BIG['libelle'] = BIG['libelle'].str.replace('site d\'injection', 'site_d_injection') 
BIG['libelle'] = BIG['libelle'].str.replace('capuchon tip-cap', 'tip-cap') 

BIG.ix[11908, 6] = BIG.ix[11908, 6].replace('25 G, 5/8', '25G, 5/8')
BIG.ix[7336, 6] = u'1 boite de 4 poche a trois compartiments (Polypropylene-co-ethylene) de 1026 ml'
BIG.ix[7804, 6] = u'1 boite de 4 poche a trois compartiments (Polypropylene-co-ethylene) de 2053 ml'
BIG.ix[9832, 6] = u'1 boite de 4 poche a trois compartiments (Polypropylene-co-ethylene) de 1540 ml'
BIG.ix[11410, 6] = u'1 flacon en verre de 150 ml avec 1 seringue avec 1 catheter'
BIG.ix[11247, 6] = u'1 flacon en verre jaune(brun) - 1 flacon en verre de 1 ml avec 1 aiguille'


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
                                     
                    
    # remove multiple spaces
    s = re.sub(r'\s+', ' ', s)
                
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

# custom replace
BIG['substances'] = substances.str.replace('UREE \[13C\]', 'UREE')


    
    
    
# FOR DEBUG
#for index, s in enumerate(BIG['substances'].sort_values().unique()):
#    print index, s


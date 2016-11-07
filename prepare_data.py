# -*- coding: UTF-8 -*-  

"""
    
    This script processes loaded data and assigns global variables with data:
    
     - PROCESSED_BIG : processed data from BIG
         - unique, no const columns, 
         
         - columns : dropped 'libelle', 'id' and 'voie admin' dummified, appended 'logprix'
         
           u'agrement col', u'date amm annee', u'date declar annee', u'etat commerc', u'forme pharma', u'libelle_ampoule',
           u'libelle_capsule', u'libelle_comprime', u'libelle_film', u'libelle_flacon', u'libelle_gelule', u'libelle_pilulier',
           u'libelle_plaquette', u'libelle_poche', u'libelle_sachet', u'libelle_seringue', u'libelle_stylo', u'libelle_tube',
           u'nb_ampoule', u'nb_capsule', u'nb_comprime', u'nb_film', u'nb_flacon', u'nb_gelule', u'nb_ml', u'nb_pilulier',
           u'nb_plaquette', u'nb_poche', u'nb_sachet', u'nb_seringue', u'nb_stylo', u'nb_tube', u'prix', u'statut', u'statut admin',
           u'titulaires', u'tx rembours', u'type proc', 'logprix', u'auriculaire', u'buccogingivale', u'cutan\xe9e', u'dentaire',
           u'endocanalaire', u'endosinusale', u'endotrach\xe9obronchique', u'gastrique', u'gastro-ent\xe9rale', u'gingivale',
           u'infiltration', u'inhal\xe9e', u'intra cholangio-pancr\xe9atique', u'intra-articulaire', u'intra-art\xe9rielle', 
           u'intra-murale', u'intra-ut\xe9rine', u'intracaverneuse', u'intradermique', u'intradurale', u'intral\xe9sionnelle', 
           u'intramusculaire', u'intrap\xe9riton\xe9ale', u'intras\xe9reuse', u'intrath\xe9cale',
           u'intraveineuse', u'intraventriculaire c\xe9r\xe9brale', u'intravitr\xe9enne', u'intrav\xe9sicale', u'nasale',
           u'ophtalmique', u'orale', u'p\xe9riarticulaire', u'p\xe9ridurale', u'p\xe9rineurale', u'p\xe9rioculaire', u'rectale',
           u'sous-cutan\xe9e', u'sublinguale', u'transdermique', u'ur\xe9trale', u'vaginale', u'voie buccale autre',
           u'voie extracorporelle autre', u'voie parent\xe9rale autre'         
         
     - PROCESSED_SUBSTANCES : dataframe with names of dummified 'substances' and an appended column '_ENCODED_'
     - PROCESSED_VOIE_ADMIN : dataframe with names of dummified 'voie admin'
     
     - TARGET_COLUMNS = ['prix', 'logprix']

"""

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.preprocessing import LabelEncoder

# Project
from load_data import train, test, BIG, TRAIN, TEST, SOURCE
from common.preprocessing_helper import get_unique_noconst, split_series_values, get_unique_col_values, get_numeric_nonnumeric_cols

# Remove some useless columns
PROCESSED_BIG = BIG.drop(['libelle', 'id'], axis=1)

# Check for dupicate rows and lets look at the columns with only one unique value.
# without targets (labels)
col_subset = PROCESSED_BIG.columns.drop('prix')
PROCESSED_BIG = get_unique_noconst(PROCESSED_BIG, col_subset, verbose=True)
PROCESSED_BIG['logprix'] = PROCESSED_BIG['prix'].apply(np.log)

TARGET_COLUMNS = ['prix', 'logprix']

# Identify and separate the numeric and non numeric rows.
num_cols, nonnum_cols = get_numeric_nonnumeric_cols(PROCESSED_BIG)

# Encode categorical features
df = PROCESSED_BIG[nonnum_cols].drop(['substances', 'voies admin', SOURCE], axis=1)
for c in df.columns:
    le = LabelEncoder()    
    le.fit(df[c])
    PROCESSED_BIG[c] = le.transform(df[c])
     
df = PROCESSED_BIG[['substances', 'voies admin']]    
    
def _alpha_sort(s):
    splt = s.split(' ')
    sorted_splt = sorted(splt)
    return ' '.join(sorted_splt) 

def _alpha_sort_notsplitted(line):
    splt = line.split(',')
    for i in range(len(splt)):
        splt[i] = _alpha_sort(splt[i])
    return ','.join(splt) 

# Substances : 
prepared_substances = df['substances'].str.strip().str.replace(', ', ',').apply(_alpha_sort_notsplitted)

PROCESSED_SUBSTANCES = prepared_substances.str.get_dummies(',')

# Check no all zero rows:
_res = PROCESSED_SUBSTANCES.sum(axis=1)
assert len(_res[_res < 1]) == 0, "There is a row with all zeros : {}".format(_res)
del _res

# Encode substances:
le = LabelEncoder()    
le.fit(prepared_substances)
PROCESSED_SUBSTANCES['_ENCODED_'] = le.transform(prepared_substances)

PROCESSED_SUBSTANCES = pd.concat([PROCESSED_SUBSTANCES, PROCESSED_BIG[SOURCE]], axis=1)


# Voies admin : 
PROCESSED_VOIE_ADMIN = PROCESSED_BIG['voies admin'].str.get_dummies(',')
PROCESSED_VOIE_ADMIN = pd.concat([PROCESSED_VOIE_ADMIN, PROCESSED_BIG[SOURCE]], axis=1)    
    
PROCESSED_BIG = PROCESSED_BIG.drop(['substances', 'voies admin'], axis=1)


                                   
                                   

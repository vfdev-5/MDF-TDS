# -*- coding: UTF-8 -*-  

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
dums_va = PROCESSED_BIG['voies admin'].str.get_dummies(',')
    
PROCESSED_BIG = PROCESSED_BIG.drop(['substances', 'voies admin'], axis=1)
PROCESSED_BIG = pd.concat([PROCESSED_BIG, dums_va], axis=1)

                                   
                                   

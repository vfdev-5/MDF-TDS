# -*- coding: UTF-8 -*-  

"""
    
    This script processes loaded data and assigns global variables with data:
    
     - PROCESSED_BIG : processed data from BIG
         - unique, no const columns, 
         
         - columns : 'id' dropped and 'libelle', 'voie admin' dummified, 'logprix' appended
         
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
     
     - PROCESSED_LIBELLES 
     - NONNUM_TYPES : [u'PULVERISATION', u'Twist-off', u'UI', u'adaptateur', u'aiguille',
                       u'ampoule', u'applicateur', u'boite', u'bouchon', u'bouteille',
                       u'calendrier', u'canule', u'capsule', u'capuchon', u'cartouche',
                       u'catheter', u'comprime', u'compte-gouttes', u'connecteur',
                       u'cuillere', u'dispositif', u'distributeur', u'dose', u'embout',
                       u'emplatre', u'fermeture', u'feuille', u'film', u'filtre',
                       u'flacon', u'g', u'gants', u'gelule', u'gobelet', u'grattoir',
                       u'implant', u'indication', u'inhalateur', u'injecteur', u'kg',
                       u'kit', u'litre', u'lyophilisat', u'materiel_de_perfusion',
                       u'mesurette', u'mg', u'ml', u'ovule', u'pailles', u'pansement',
                       u'pilule', u'pilulier', u'pinceau', u'pipette', u'plaquette',
                       u'poche', u'pochette', u'pompe', u'pot', u'prolongateur',
                       u'protege-aiguille', u'racleur', u'recipient', u'sac', u'sachet',
                       u'sachet-dose', u'semainier', u'seringue',
                       u'site', u'sparadraps', u'spatule', u'stilligoutte', u'stylo',
                       u'suppositoire', u'systeme_de_recuperation', u'systeme_de_securite',
                       u'tampon', u'tip-cap', u'tube', u'unites', u'valve']

     - MEASURE_TYPES : [u'UI', u'mg', u'ml', u'g', u'kg', u'litre', u'PULVERISATION']
     
     Types do not containing substances :
     - NON_QUANTITY_TYPES = [
                                u'Twist-off', u'adaptateur', u'aiguille', u'applicateur', u'bouchon', u'calendrier', u'canule',
                                u'catheter', u'compte-gouttes', u'connecteur', u'distributeur', u'embout', u'emplatre', 
                                u'fermeture', u'feuille', u'filtre', u'gants', u'gobelet', u'grattoir', u'indication', u'injecteur',
                                u'kit', u'materiel_de_perfusion', u'pansement', u'pinceau', u'pompe', u'prolongateur',
                                u'protege-aiguille', u'racleur', u'securite', u'semainier', u'site_d_addition', u'site_d_injection',
                                u'sparadraps', u'spatule', u'stilligoutte', u'systeme_de_recuperation', u'systeme_de_securite',
                                u'tampon', u'tip-cap', u'valve' 
                            ]
     
     Types do containing substances :
     - QUANTITY_TYPES = [plaquettes, doses, etc ]
     
     - NB_COLS : nb_*
     - LIBELLE_COLS : libelle_*
     
     - TARGET_COLUMNS = ['prix', 'logprix']

"""
import re
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
PROCESSED_BIG = BIG.drop(['id'], axis=1)

# Check for dupicate rows and lets look at the columns with only one unique value.
# without targets (labels)
col_subset = PROCESSED_BIG.columns.drop('prix')
PROCESSED_BIG = get_unique_noconst(PROCESSED_BIG, col_subset, verbose=True)
PROCESSED_BIG['logprix'] = PROCESSED_BIG['prix'].apply(np.log)

TARGET_COLUMNS = ['prix', 'logprix']

# Parse 'libelle'
_libelles = PROCESSED_BIG.ix[:, 5].values

MEASURE_TYPES = [u'UI', u'mg', u'ml', u'g', u'kg', u'litre', u'PULVERISATION']
NONNUM_TYPES = []
_data = []
# First pass : get all types of packages, quantity types etc 
for t in _libelles:
    tt = t
    ## Add '1' before package type 
    if not tt[0].isdigit():
        tt = '1 ' + tt   
    
    ## Add 1 between 'avec' and a word after
    groups = re.findall("(avec) ([a-zA-Z]+)", t)
    for g in groups:
        init = ' '.join(g)
        repl = '%s 1 %s' % g
        tt = tt.replace(init, repl)
    
    ## Get groups (digit, type)
    groups = re.findall(r'(\d+,?\d*)\s+([\w\-]+)', tt)
    if len(groups) > 0:
        for g in groups:
            NONNUM_TYPES.append(g[1])
        _data.append(groups)
    else:
        raise Exception("Not found : %s" % t)

del _libelles       
NONNUM_TYPES = np.unique(np.array(NONNUM_TYPES))

NON_QUANTITY_TYPES = [
    u'Twist-off', u'adaptateur', u'aiguille', u'applicateur', u'bouchon', u'calendrier', u'canule',
    u'catheter', u'compte-gouttes', u'connecteur', u'cuillere', u'distributeur', u'embout', u'emplatre', 
    u'fermeture', u'feuille', u'filtre', u'gants', u'gobelet', u'grattoir', u'indication', u'injecteur',
    u'kit', u'materiel_de_perfusion', u'mesurette', u'pansement', u'pinceau', u'pipette', u'pompe', u'prolongateur',
    u'protege-aiguille', u'racleur', u'securite', u'semainier', u'site_d_addition', u'site_d_injection',
    u'sparadraps', u'spatule', u'stilligoutte', u'systeme_de_recuperation', u'systeme_de_securite',
    u'tampon', u'tip-cap', u'valve' 
]

QUANTITY_TYPES = list(set(NONNUM_TYPES) - set(NON_QUANTITY_TYPES) - set(MEASURE_TYPES))

# Second pass : 
processed_libelles = np.zeros((len(_data), len(NONNUM_TYPES)))
PROCESSED_LIBELLES = pd.DataFrame(processed_libelles, columns=NONNUM_TYPES, index=PROCESSED_BIG.index)    
for index, groups in zip(PROCESSED_BIG.index, _data):
    for g in groups:
        if g[1] in MEASURE_TYPES:
            # Write the biggest measure type 
            val = float(g[0].replace(',','.'))
            if PROCESSED_LIBELLES.ix[index, g[1]] < val:
                PROCESSED_LIBELLES.ix[index, g[1]] = val
        else:
            PROCESSED_LIBELLES.ix[index, g[1]] = g[0]
        
PROCESSED_LIBELLES = pd.concat([PROCESSED_LIBELLES, PROCESSED_BIG[SOURCE]], axis=1)
        
del _data        
        
# Identify and separate the numeric and non numeric rows.
num_cols, nonnum_cols = get_numeric_nonnumeric_cols(PROCESSED_BIG)

# Encode categorical features
df = PROCESSED_BIG[nonnum_cols].drop(['libelle', 'substances', 'voies admin', SOURCE], axis=1)
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

# COLUMNS
NB_COLS = [u'nb_ampoule', u'nb_capsule', u'nb_comprime', u'nb_film', u'nb_flacon', u'nb_gelule', u'nb_ml', u'nb_pilulier', u'nb_plaquette', u'nb_poche', u'nb_sachet', u'nb_seringue', u'nb_stylo', u'nb_tube']

LIBELLE_COLS = [u'libelle_ampoule', u'libelle_capsule', u'libelle_comprime', u'libelle_film', u'libelle_flacon', u'libelle_gelule', u'libelle_pilulier', u'libelle_plaquette', u'libelle_poche', u'libelle_sachet', u'libelle_seringue', u'libelle_stylo', u'libelle_tube']
                                   


        
        
        
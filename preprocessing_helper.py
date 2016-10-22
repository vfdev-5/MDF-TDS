
"""
    Helper methods to preprocess data : cleaning, feature engineering
    
"""

import numpy as np
import pandas as pd


def drop_const_cols(df):
    """
    Method to remove constant columns
    """
    return df.loc[:, (df != df.ix[0]).any()]


def get_unique_noconst(df, verbose=False):
    """
    Method returns a DataFrame without duplicated rows and constant columns
    """
    if verbose:
        print "-- get_unique_noconst --"
    duplicated = df.duplicated()
    nb_duplicates = np.sum(duplicated)
    if verbose:        
        print "- Number of found duplicated rows : ", nb_duplicates
        
    if nb_duplicates > 0:
        df_unique = df.drop_duplicates()
    
    df_unique_noconst = drop_const_cols(df_unique)
    if verbose:
        print "- Remove constant columns : ", df_unique.shape, '->', df_unique_noconst.shape
        print "- Dropped const columns : ", 
        if len(df_unique.columns) > len(df_unique_noconst.columns):
            print df_unique.columns.difference(df_unique_noconst.columns).values
    
    return df_unique_noconst

            
    
    



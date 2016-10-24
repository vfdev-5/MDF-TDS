
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


def get_numeric_nonnumeric_cols(df):
    """
    Method returns two lists with numeric and nonnumeric colum names
    """
    num_cols = []
    nonnum_cols = []
    type_groups = df.columns.to_series().groupby(df.dtypes)
    for key in type_groups.groups:
        if np.issubdtype(key, np.number):
            num_cols.extend(type_groups.get_group(key).values)
        else:
            nonnum_cols.extend(type_groups.get_group(key).values)
    
    return num_cols, nonnum_cols
    
            
def split_series_values(series, splitter=','):
    """
    Method returns a pd.Series of splitted input values given as pd.Series
    Input should be non-numerical
    For example, ['a', 'a,b', 'b', 'a,b,c'] -> ['a', 'a', 'b', 'b', 'a', 'b', 'c']
    """
    res = series.apply(lambda val: val.split(splitter)).values
    ret = []
    for a in res:
        ret.extend(a)
    return pd.Series(ret)    
    

def get_unique_col_values(df, display_nb_cols=10):
    """
    Method returns a DataFrame with columns and their unique values.
    Use for non-numeric columns only.
    
    For example, input dataframe :
        A    B    C 
        v1   w1   u1
        v1   w2   u2
        v2   w2   u3
        v3   w2   u3
    
    Output dataframe is :
    
        A  3  v1    v2    v3
        B  2  w1    w2
        C  3  u1    u2    u3
        
    """
    nonnum_cols = df.columns
    max_nb_cols = 0
    rows = []
    for i, col_name in enumerate(nonnum_cols):
        ll = df[col_name].unique().tolist()
        row = [col_name, len(ll)]
        row.extend(ll)
        max_nb_cols = max(max_nb_cols, len(row))
        rows.append(row)
        
    out_df = pd.DataFrame(columns=range(max_nb_cols), index=range(len(nonnum_cols)))    
    for i in range(len(out_df)):
        row = rows[i]
        if len(row) < max_nb_cols:
            row.extend([""]*(max_nb_cols - len(row)))
        out_df.iloc[i] = row
    return out_df

"""
    Helper methods to visualize data
"""

import numpy as np
import pandas as pd


def display_groups(groupby_object):
    """
    Display pandas GroupBy object as 
    -- key1 --
    
        value1
        value2
        ...
        
    -- key2 --
    
        value1
        value2
        ...
     ...
    """
    for key, value in groupby_object.groups.items():
        print '\n--', key, '--'
        print "\n\t", "\n\t".join(value)


def display_unique_count(series):
    """
    Method returns a DataFrame to display all unique column values
    and their count. Output is sorted in descending order.  
    Use for non-numeric columns only.
    
    For example, input dataframe :
        A    B    C 
        v1   w1   u1
        v1   w2   u2
        v2   w2   u3
        v3   w2   u3
    
    Output dataframe is :
    
        Values    Count
          w2        3
          v1        2
          u3        2
          v2        1 
          v3        1
          w1        1
          u1        1
    """
    data = np.unique(series, return_counts=True)
    df = pd.DataFrame(
        data = {'Value':data[0], 'Count':data[1]},
        columns = ['Value', 'Count'])
    df.sort_values(by ="Count", ascending=False, inplace=True)
    return df


def display_unique_col_values(df, display_nb_cols=10):
    """
    Method returns a DataFrame to display columns and their unique values.
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

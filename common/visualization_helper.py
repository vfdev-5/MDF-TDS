
"""
    Helper methods to visualize data
"""

import numpy as np
import pandas as pd

# Project 


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
  

"""
    Helper methods to visualize data
"""

import numpy as np
import pandas as pd

def display_groups(groupby_object):
    for key, value in groupby_object.groups.items():
        print '\n--', key, '--'
        print "\n\t", "\n\t".join(value.values)


def display_unique_count(series):
    data = np.unique(series, return_counts=True)
    df = pd.DataFrame(
        data = {'Value':data[0], 'Count':data[1]},
        columns = ['Value', 'Count'])
    df.sort_values(by ="Count", ascending=False, inplace=True)
    return df
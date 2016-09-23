import pandas as pd
import numpy as np

def indices_seperate(feature_name=None, data_frame=pd.DataFrame()):
    '''
        Indices at value changing points, For one dimensional array

        :param feature_name: Name of the column
        :param data_frame: Pandas data frame
        :return: Inices array
    '''

    column = []
    if not feature_name:
        print "Feature name is empty"
        return

    try:
        column = data_frame[feature_name]
    except KeyError:
        # Key not found exception
        print "Key not found"
        return

    if len(column) == 0:
        # There is nothing to slice
        print "Nothing to slice"
        return

    column = np.array(column)

    # Each index where the value changes
    indices = np.where(column[:-1] <> column[1:])[0]
    return indices

def slice(data_column=np.array([]), indices=np.array([])):
    '''
        Slice a given array according to the indices

        :param data_column: One dimensional array
        :param indices: indices to slice
        :return: sliced list
    '''

    # Add starting point and adjust indices
    indices = np.insert(indices+1, 0, 0, axis=0)
    # Add ending point
    indices = np.insert(indices, len(indices),len(data_column), axis=0)

    seperated_list = []
    for i in range(len(indices)-1):
        seperated_list.append(data_column[indices[i]:indices[i+1]])

    return seperated_list



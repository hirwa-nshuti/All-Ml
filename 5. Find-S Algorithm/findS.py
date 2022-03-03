import pandas as pd
import numpy as np


# Loading the dataSet
def read_data(filename):
    """
    After data preparation this function returns the independent variables and dependent variables
    for EnjoySports DataSet from Tom M. Mitchel's Machine Learning book
    args: filename the data csv file to be loaded
    """
    data = pd.read_csv(filename)
    df = np.array(data)[:, :-1]
    target = np.array(data)[:, -1]

    return df, target


# The training function for find-S algorithm
def train(c, targ):
    """
    Function that returns the generalized hypothesis from given training data
    :param c: contains all specific hypothesis
    :param targ: The target variable of the data
    :return: generalized hypothesis
    """
    for i, val in enumerate(targ):
        if val == "Yes":
            h = c[i].copy()
            break
    for i, val in enumerate(c):
        if targ[i] == "Yes":
            for x in range(len(h)):
                if val[x] != h[x]:
                    h[x] = '?'
        else:
            pass
    return h


if __name__ == "__main__":
    file = "EnjoySport.csv"
    c, targ = read_data(file)
    print(f"The final generalized hypothesis is {train(c, targ)}")

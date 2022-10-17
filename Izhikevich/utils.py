import os
import datetime
import time
import random
import numpy as np
def check_paths(dir_path):
    #check if a dir path exist, otherwise create it.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_datetime():
    #get timestamp(int to str)
    return time.ctime().replace(" ", "-")

def sample_matrix_axon2(mat,k):
    sample_row_list = random.sample(range(mat.shape[1]),k)
    result = np.vstack(np.array([list(mat[:,s]) for s in sample_row_list]))
    result = result.T
    return result


def spike_to_coordinate(ts,sp_matrix):
    # sp_matrix = np.asarray(sp_matrix)
    # ts = np.asarray(ts) 
    # get index and time
    elements = np.where(sp_matrix)
    print(elements[0])
    print(elements[1])
    index = np.asarray([element for element in elements[1]])
    time = np.asarray([ts[element] for element in elements[0]]) 
    return time, index

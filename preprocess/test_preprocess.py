import numpy as np
from sklearn.decomposition import PCA
import pickle
import random

with open('processing/ttoi', 'rb') as f:
    ttoi = pickle.load(f)
stored_teams = list(ttoi.keys())

with open('processing/ptoi', 'rb') as f:
    ptoi = pickle.load(f)
stored_pitchers = list(ptoi.keys())

with open('processing/pcacolumns', 'rb') as f:
    pcacolumns = pickle.load(f)

with open('processing/pca_processors', 'rb') as f:
    pca_processors = pickle.load(f)

def convert_team(team_name: str):
    if team_name in stored_teams:
        return ttoi[team_name]
    return 0.5

def convert_pitcher(pitcher_name: str):
    if pitcher_name in stored_pitchers:
        return ptoi[pitcher_name]
    return 0.5

def preprocess_data(test_data: np.ndarray, stage=1):
    print("start preprocessing")
    data = np.delete(test_data, [0, 37, 38], axis=1)
    x, y = data.shape
    
    print("modifying and filling data")
    for i in range(x):
        data[i][0] = convert_team(data[i][0]) # convert team to probabilities
        data[i][1] = convert_team(data[i][1])
        data[i][2] = 1 if data[i][2] else 0 # convert true false to (1, 0)
        data[i][3] = convert_pitcher(data[i][3]) # convert pitcher to probabilities
        data[i][4] = convert_pitcher(data[i][4])
        if stage == 1:
            data[i][9] = data[i][9]-2016 if data[i][9] > 2015 else random.randint(0, 7) # modify seasons
        else:
            data[i][9] = 8
        for j in range(5, y): # convert nan to 0
            if (not data[i][j] > 0) and (not data[i][j] < 0):
                data[i][j] = 0
    
    print("transforming data")
    transformed_data = []
    to_be_deleted = []
    num_pca = len(pca_processors)
    
    for i in range(num_pca):
        selected_data = data[:, pcacolumns[i]] # get data to be transformed
        transformed_data.append(pca_processors[i].transform(selected_data)) # transform by stored pca functions
        to_be_deleted = to_be_deleted + pcacolumns[i] # record columns to be deleted
        
    processed_data = np.delete(data, to_be_deleted, axis=1) # delete original columns
    
    for i in range(num_pca):
        processed_data = np.hstack((processed_data, transformed_data[i])) # concatenate transformed columns to data
    return processed_data

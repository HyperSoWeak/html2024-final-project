import numpy as np
from sklearn.decomposition import PCA
import pickle
import random
from scipy.stats import pearsonr
import os
from tqdm import tqdm


if not os.path.exists('processing'):
    os.makedirs('processing')


class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    
    def get_sets(self):
        sets = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in sets:
                sets[root] = []
            sets[root].append(i)
        return sets


def create_dictionary(names: np.ndarray, y: np.ndarray):
    name_set = set() # find all unique names
    for i in names:
        for j in i:
            name_set.add(j)
    
    dictionary = {}
    for i in name_set: # initialize dictionary [win plays, total plays]
        dictionary[i] = [0, 0]
    for i in range(names.shape[0]):
        dictionary[names[i][0]][1] += 1 # counting plays
        dictionary[names[i][1]][1] += 1
        if y[i]: # home team wins
            dictionary[names[i][0]][0] += 1
        else: # away team wins
            dictionary[names[i][1]][0] += 1
    
    newdict = {}
    for i in range(names.shape[0]):
        for j in range(2): # calculate winning probability
            newdict[names[i][j]] = dictionary[names[i][j]][0]/dictionary[names[i][j]][1]
    return newdict

def preprocess_data(train_data: np.ndarray, r2_threshold=0.25, plots_path='preprocess/processing/plots', output_path='preprocess/processing', dataset_name='processed_data', no_plot=True):
    print("start preprocessing")
    ground_truth = train_data[:, 5] # isolate ground truth out of training data
    data = np.delete(train_data, [0, 3, 5, 39, 40], axis=1) # remove unnecessary columns
    ttoi = create_dictionary(data[:, [0, 1]], ground_truth) # map team to probabilities
    ptoi = create_dictionary(data[:, [3, 4]], ground_truth) # map pitcher to probabilities
    
    print("mapping and filling data")
    x, y = data.shape
    for i in range(x):
        data[i][0] = ttoi[data[i][0]] # convert team to probabilities
        data[i][1] = ttoi[data[i][1]]
        data[i][2] = 1 if data[i][2] else 0 # convert true false to (1, 0)
        data[i][3] = ptoi[data[i][3]] # convert pitcher to probabilities
        data[i][4] = ptoi[data[i][4]]
        data[i][9] = data[i][9]-2016 if data[i][9] > 2015 else random.randint(0, 7) # modify seasons
        for j in range(5, y): # convert nan to 0
            if (not data[i][j] > 0) and (not data[i][j] < 0):
                data[i][j] = 0
    
    print("calculating r values")
    r_values = np.zeros((y, y)) # calculate r value for each column
    for i in tqdm(range(10, y)):
        for j in tqdm(range(10, y), leave=False):
            r_values[i][j] = float(pearsonr(data[:, i].astype(float), data[:, j].astype(float))[0])**2
    
    r_masks = np.logical_and(r_values > r2_threshold, r_values < 0.99)
    related_pairs = []
    for i in range(10, y):
        for j in range(10, y):
            if r_masks[i][j] and ((i, j) not in related_pairs and (j, i) not in related_pairs):
                related_pairs.append((i, j))
    
    pairs = len(related_pairs)
    
    if not no_plot:
        print("plotting\n")
        import matplotlib.pyplot as plt
        
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        blue = np.copy(ground_truth)
        blue = blue.astype(bool)
        green = np.logical_not(blue)
        for i in range(pairs):
            px, py = related_pairs[i]
            plt.scatter(data[blue, px], data[blue, py], c='blue', label='home wins', alpha=0.2)
            plt.scatter(data[green, px], data[green, py], c='green', label='home losses', alpha=0.2)
            plt.title(f'{px} vs {py}, r={r_values[px][py]:.2f}')
            plt.xlabel(f'column {px}')
            plt.ylabel(f'column {py}')
            plt.legend(loc='best')
            plt.savefig(f'{plots_path}/{r_values[px][py]:.2f}r_{px}_to_{py}.png')
            plt.close()
    
    dsu = DisjointSetUnion(y)
    for px, py in related_pairs:
        dsu.union(px, py)
    
    print("applying PCA on:")
    sets = dsu.get_sets()
    pcacolumns = []
    for _, item in sets.items():
        if len(item) > 1:
            print(item)
            pcacolumns.append(item)
    
    print("\nstart PCA")
    pca_processors = [PCA(n_components=int(len(i)/2)) for i in pcacolumns]
    transformed_data = []
    to_be_deleted = []
    num_pca = len(pca_processors)
    
    for i in range(num_pca):
        selected_data = data[:, pcacolumns[i]]
        transformed_data.append(pca_processors[i].fit_transform(selected_data))
        to_be_deleted = to_be_deleted + pcacolumns[i]
    
    processed_data = np.delete(data, to_be_deleted, axis=1)
    
    print("saving necessary files")
    
    for i in range(num_pca):
        processed_data = np.hstack((processed_data, transformed_data[i]))
    
    with open(f'{output_path}/ttoi', 'wb') as f:
        pickle.dump(ttoi, f)
    
    with open(f'{output_path}/ptoi', 'wb') as f:
        pickle.dump(ptoi, f)
    
    with open(f'{output_path}/pcacolumns', 'wb') as f:
        pickle.dump(pcacolumns, f)
    
    with open(f'{output_path}/pca_processors', 'wb') as f:
        pickle.dump(pca_processors, f)
    
    with open(f'{output_path}/ground_truth', 'wb') as f:
        pickle.dump(ground_truth, f)
    
    with open(f'{output_path}/{dataset_name}', 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data

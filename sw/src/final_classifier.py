# installing dependencies
import mne
#from mne.decoding import CSP
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def seven_bandpass(X):
    ''' 
    filter#:  Hz
    filter0: 4-8
    filter1: 8-12
    filter2: 12-16
    filter3: 16-20
    filter4: 20-24
    filter5: 24-28
    filter6: 28-32
    '''
    fR = [i*4 for i in range(1, 9)]

    filtered = []
    for i in range(7):
        filtered.append(X.copy().filter(fR[i],fR[i+1]))
    return np.array(filtered)

def seven_pca_transform(X_seven_F, pca_list):
    X_seven_t = []
    for i in range(7):
        X_seven_t.append(pca_list[i].transform(X_seven_F[i]))
    return np.array(X_seven_t)

def seven_svm_transform(X,svm_l):
    X_seven_t = []
    for i in range(7):
        X_seven_t.append(svm_l[i].predict(X[i,:,:]))
    return np.array(X_seven_t)

def seven_svm_votes(votes):
    """
    Given a numpy vector, return the most common value.
    """
    values, counts = np.unique(votes, return_counts=True)
    most_common_value = values[np.argmax(counts)]
    return most_common_value
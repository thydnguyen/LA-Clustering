import sklearn
from numba import jit
import sys 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import random

from geom_median.numpy  import compute_geometric_median

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# return k means cost given centers
def k_means_cost(points, centers):
    distance = euclidean_distances(points, centers)
    distance = distance**2
    labels = np.argmin(distance, axis=1)
    return labels, np.min(distance, axis = 1).sum()

def k_means_labels(points, centers):
    distance = euclidean_distances(points, centers)
    distance = distance**2
    labels = np.argmin(distance, axis=1)
    return labels

# return k means cost given labels of points
def kmeans_cost_label(points, labels, num_labels):
    _ , d = np.shape(points)
    centers = np.zeros((num_labels, d))
    good_indices = []
    for i in range(num_labels):
        to_index = np.where(labels == i)[0]
        if len(to_index) > 0:
            curr_points = points[to_index]
            centers[i,:] = np.average(curr_points, axis = 0)
            good_indices.append(i)
        else:
            pass
    centers = centers[good_indices,:]
        
    return k_means_cost(points, centers)

# return k means cost given centers
def k_medians_cost(points, centers):
    distance = euclidean_distances(points, centers)
    labels = np.argmin(distance, axis=1)
    return labels, np.min(distance, axis = 1).sum()

def k_medians_labels(points, centers):
    distance = euclidean_distances(points, centers)
    labels = np.argmin(distance, axis=1)
    return labels

# return k means cost given labels of points
def kmedians_cost_label(points, labels, num_labels):
    _ , d = np.shape(points)
    centers = np.zeros((num_labels, d))
    good_indices = []
    for i in range(num_labels):
        to_index = np.where(labels == i)[0]
        if len(to_index) > 0:
            curr_points = points[to_index]
            centers[i,:] = compute_geometric_median(curr_points).median
            good_indices.append(i)
        else:
            pass
    centers = centers[good_indices,:]
        
    return k_medians_cost(points, centers)

    
def hard_noisy_oracle(data, label, prob_error):
    
    num_labels = len(np.unique(label))
    new_label = np.copy(label)
    for i in range(num_labels):
        to_index = np.where(label == i)[0]
        numCorrupt = int(np.floor(prob_error * len(to_index)))
        mean = np.mean(data[to_index], axis = 0, keepdims=True)
        dist = euclidean_distances(mean, data[to_index])
        index_to_remove = np.argsort(dist)[0,:numCorrupt]
        #print(index_to_remove)
        randChoice = list(range(0, i)) + list(range(i+1, num_labels))
        #print(np.shape(new_label[index_to_remove] ))
        new_label[to_index[index_to_remove]] = np.random.choice(randChoice, size = numCorrupt)
    #print(sum(new_label != label)/len(label))
    return new_label

def hard_noisy_oracle_median(data, label, prob_error):
    
    num_labels = len(np.unique(label))
    new_label = np.copy(label)
    for i in range(num_labels):
        to_index = np.where(label == i)[0]
        numCorrupt = int(np.floor(prob_error * len(to_index)))
        median = [compute_geometric_median(data[to_index]).median]
        dist = euclidean_distances(median, data[to_index])
        index_to_remove = np.argsort(dist)[0,:numCorrupt]
        #print(index_to_remove)
        randChoice = list(range(0, i)) + list(range(i+1, num_labels))
        #print(np.shape(new_label[index_to_remove] ))
        new_label[to_index[index_to_remove]] = np.random.choice(randChoice, size = numCorrupt)
    return new_label


# faster version of algorithm 2 using numba jit, tested with this version
@jit(nopython=True)
def algo2new(points, eps):
    
    n = len(points)
    
    if n <= 10:
        return points.mean()
    
    to_return = 0.0
    for i in range(1):
        points = np.random.permutation(points)
        X1 = points[:n//2]
        X2 = points[n//2:]
        X1 = np.sort(X1)

        counter = int((1-5*eps)*(n//2))

        
        if counter == 1:
            to_return += X2.mean()
        else:
            X1_left = X1[:-counter+1]
            X1_right = X1[counter-1:]

            good_indx = np.argmin(X1_right-X1_left)
            a = X1_left[good_indx]
            b = X1_right[good_indx]
            to_index = np.where((a <= X2) & (X2 <= b))[0]
            if len(to_index) == 0:
                to_return += 0.0
            else:
                to_return += X2[to_index].mean()

    return to_return/1
    


# algorithm 2 from paper without jit, not tested
def algo2(points, eps):
    n = len(points)
    
    to_return = 0.0
    if n < 10:
        return sum(points)/n

    for i in range(25):
        # randomly partition points into two groups of equal size
        points = np.random.permutation(points)
        X1 = points[:n//2]
        X2 = points[n//2:]
        X1 = np.sort(X1)

        # find interval of X1 with (1-eps) fraction of points
        # call this interval [a,b]
        counter = int((1-5*eps)*(n//2))
        curr_len = float('inf')
        a = 0
        b = 0
        for i in range(n//2-counter+1):
            curr_int_left = X1[i]
            curr_int_right = X1[i+counter-1]
            if curr_int_right -  curr_int_left < curr_len:
                a = curr_int_left
                b = curr_int_right
                curr_len = b - a
        X2_filtered = [x for x in X2 if a <= x <= b]


        # return average of points in X2 that are in [a,b]
        if len(X2_filtered) == 0:
            to_return += 0.0
        else:
            to_return += sum(X2_filtered)/len(X2_filtered)
    return to_return/25.0

# main algo of paper
def algo1(points, oracle_labels, k, eps):
    n,d = points.shape
    centers = np.zeros((k, d))
    labels_so_far = []

    # loop over each label
    for i in range(k):

        # get labels that haven't been processed so far
        good_indices = np.where(~np.isin(oracle_labels, labels_so_far))[0]
        curr_labels = oracle_labels[good_indices]
        
        if len(curr_labels) > 0:

            # get most common label
            label_counts = np.bincount(curr_labels)
            most_common_label = np.argmax(label_counts)
            points_with_labels = points[np.where(oracle_labels == most_common_label)[0]]


            # for most common label, loop over each dimension and run alg 2
            for j in range(d):
                curr_dim_points = points_with_labels[:,j]
                curr_dim_center = algo2new(curr_dim_points, eps)
                centers[most_common_label, j] = curr_dim_center
            
            labels_so_far.append(most_common_label)

                
        else:
            pass
    return centers



#sampling baseline
def sampling_baseline(points, labels,  num_labels, rate = 50):
    n,d = np.shape(points)
    rate = rate/100.0
    centers = np.zeros((num_labels, d))
    good_indices = []
    for i in range(num_labels):
        to_index = np.where(labels == i)[0]
        size_to_keep = int(rate*len(to_index))
        if size_to_keep > 0:
            to_index = np.random.permutation(to_index)
            curr_points = points[to_index[:size_to_keep]]
            centers[i,:] = np.average(curr_points, axis = 0)
            good_indices.append(i)
        else:
            pass
    centers = centers[good_indices,:]
    return k_means_cost(points, centers)

def samplingResult(points, labels,  num_labels):
    sampling_cost = np.inf
    for i in range(1, 51):
        curr_error = sampling_baseline(points, labels, num_labels, i)[1]
        if curr_error < sampling_cost:
            sampling_cost = curr_error
    return sampling_cost

#sampling baseline
def sampling_baseline_medians(points, labels,  num_labels, rate = 50):
    n,d = np.shape(points)
    rate = rate/100.0
    centers = np.zeros((num_labels, d))
    good_indices = []
    for i in range(num_labels):
        to_index = np.where(labels == i)[0]
        size_to_keep = int(rate*len(to_index))
        if size_to_keep > 0:
            to_index = np.random.permutation(to_index)
            curr_points = points[to_index[:size_to_keep]]
            centers[i,:] = compute_geometric_median(curr_points).median
            good_indices.append(i)
        else:
            pass
    #print(len(good_indices))
    centers = centers[good_indices,:]
    return k_medians_cost(points, centers)

def samplingResultMedians(points, labels,  num_labels):
    sampling_cost = np.inf
    for i in range(1, 51):
        curr_error = sampling_baseline_medians(points, labels, num_labels, i)[1]
        if curr_error < sampling_cost:
            sampling_cost = curr_error
    return sampling_cost
            


def detAlg(points, oracle_labels, k, eps):
    n,d = points.shape
    centers = np.zeros((k, d))

    # loop over each label
    for i in range(k):
        points_with_labels = points[np.where(oracle_labels == i)[0]]
        for j in range(d):
            curr_dim_points = points_with_labels[:,j]
            curr_dim_center = smallCluster(curr_dim_points, eps)
            centers[i, j] = curr_dim_center
    return centers

@jit(nopython=True)
def smallCluster(L, eps):
    K = int(np.floor(len(L)*(1-eps)))
    L = np.sort(L)
    S = np.sum(L[:K])
    S_square = np.sum((L**2)[:K])    
    best_mean = S / K
    best_cost = S_square - S**2 / K
    costList = []
    costList.append(best_cost)
    for i in range(K, len(L)):
        S_square += L[i]**2 - L[i-K]**2 
        S += L[i] - L[i-K] 
        cost_all = S_square - S**2 / (K)
        costList.append(cost_all)
        if cost_all < best_cost:
            best_mean = S/K
            best_cost = cost_all

    return best_mean

def algo1Medians(points, oracle_labels, k, eps, iterN = 1):
    n,d = points.shape
    centers = np.zeros((k, d))
    sampleN = int(1/eps**4 * (np.log(k/ eps))**2)

    # loop over each label
    for i in range(k):
        points_with_labels = np.where(oracle_labels == i)[0]
        if len(points_with_labels) < sampleN:
            #print("not enogh points for", k)
            #print(points[points_with_labels])
            centers[i] =  compute_geometric_median(points[points_with_labels]).median
        else:
            best = float('inf')
            for j in range(iterN):
                randSubset = points[np.random.choice(points_with_labels, sampleN)]
                gm = compute_geometric_median(randSubset).median
                if iterN > 1:
                    cost = k_medians_cost(points[points_with_labels], [gm])[1]
                    if cost < best:
                        best = cost
                        centers[i] = gm
                else:
                    centers[i] = gm
                    
    return centers

def algo2Medians(points, oracle_labels, k, eps, iterN = 1):

    n,d = points.shape
    centers = np.zeros((k, d))
    # loop over each label
    for i in range(k):
        points_with_labels = np.where(oracle_labels == i)[0]
        m_i = len(points_with_labels)
        best = float('inf')
        for j in range(iterN):
            randPoint = np.random.choice(points_with_labels)
            randDist = euclidean_distances(points[[randPoint]], points[points_with_labels])
            
            index_to_keep = np.argsort(randDist)[0,: int((1-eps) * m_i)]
            #print(np.argsort(randDist))
            #print(len(index_to_keep), len(points_with_labels))
            randSubset = points[points_with_labels[index_to_keep]]
            gm = compute_geometric_median(randSubset).median
            #print(np.average(randSubset, axis = 0))
            if iterN > 1:
                
                cost = k_medians_cost(points[points_with_labels], [gm])[1]
                if cost < best:
                    best = cost
                    centers[i] = gm
            else:
                centers[i] = gm

    return centers
    
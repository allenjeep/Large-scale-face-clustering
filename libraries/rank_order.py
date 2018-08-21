import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
import os
import sys
import shutil

# rank order clustering
# igonore labels just feature as input 64x64 #samples === 4069 *#samples
# KNN  input should be matrix with len of sample and row len of resolution(4069)
# this does not consider the parameter setting of svm, the soft margin error
# instead of LIBLINEAR library we use sklearn svm, C parameter =10
# further improvement to be done  threshold need to be decided


# Constants
PIC_SIZE = 4096
SAMPLE_SIZE = 10
NEIGHBOUR_SIZE = 10
K = 7 #nearst neighbor
N1 = 45
N2 = 55 #range of negative node  in paper it can be N1=50 N2=60

# Functions
def data_create(k):
    d = np.random.random_sample((k,PIC_SIZE))
    return d

#KNN neighbour ,return matrix with data_id and its neighbour , self included
def neighbour_k(object,k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(object)
    distances, indices = nbrs.kneighbors(object)
    return indices
	
def similarity_matrix(size, data, neighbour):
	# build similarity matrix
	s_matrix = np.zeros((size,size))
	for i in range(size):
		for j in range(size):
			s_matrix[i][j] = neighbour_similarity(i, j, data, neighbour)
			
	# symetric
	tmp_s = np.transpose(s_matrix)
	s_matrix = (tmp_s + s_matrix) / 2
	
	return s_matrix

#build new data with only necessary for svm, K is nearst neighbour , N1 N2 range for negative
# return new data and its label matrix , 1 for positive, -1 for negative
def svm_data_label(v, data, neighbour):
    length = K + N2 - N1
    new_data = np.zeros((length, PIC_SIZE))
    for i in range(K):
        new_data[i] = data[neighbour[v][i]]
    for i in range(N1, N2):
        new_data[K + i - N1] = data[neighbour[v][i]]
    label = np.ones(len(new_data))
    for i in range(K, len(label)):
        label[i] = -1
    return new_data, label

# SVM input is neighbour and far away point of vector, output is weight and intecept
def hyperplane(v, data, neighbour):
    clf = LinearSVC(random_state=0, C=10)
    clf.fit(svm_data_label(v, data, neighbour)[0],svm_data_label(v, data, neighbour)[1])
    w = clf.coef_
    b = clf.intercept_
    return w, b

# similarity asyemtric v1,v2 is the id
def svm(v1, v2, data, neighbour):
    inner_p = np.dot(hyperplane(v1, data, neighbour)[0], data[v2]) + hyperplane(v1, data, neighbour)[1]
    return inner_p

#neighbour group similarity
def neighbour_similarity(v1, v2, data, neighbour):
    s = 0
    for i in range(K): # consider KNN of v2
        s = s + svm(v1, neighbour[v2][i], data, neighbour)
    return s / K #average
	
def get_biggest_cluster(clustering):
	unique_clusters = np.unique(clustering)
	result = list()
	biggest = 0
	
	for c in range(len(unique_clusters)):
		cluster = list()
		for i in range(len(clustering)):
			if clustering[i] == unique_clusters[c]:
				cluster.append(i)
		result.append(cluster)
		if len(result[c]) > len(result[biggest]):
			biggest = c
		
	return result

def test_random_floats():
	data = data_create(SAMPLE_SIZE)
	cluster(data)
	
def cluster(data, input, output):
	SAMPLE_SIZE = len(data)
	NEIGHBOUR_SIZE = len(data)
	PIC_SIZE = len(data[0])
	neighbour = neighbour_k(data, NEIGHBOUR_SIZE) # for labeling later, need large sequence because of negative
	s_matrix = similarity_matrix(len(data), data, neighbour)

	# bounded  from similarity to distance
	d_matrix = np.exp(np.multiply(s_matrix, -1))

	# make diagonal equal to 0
	d_matrix = d_matrix - np.multiply(d_matrix, np.eye(len(data)))

	# clustering , threshold still need to be considered, in paper 2.3
	# sklearn does not have distance threshold parameter , use scipy instead
	# convert to condensed d_matrix
	d_matrix_cond = squareform(d_matrix)
	Z = hcluster.linkage(d_matrix_cond, method='average')
	
	#fcluster take matrix from linkage
	clustering = hcluster.fcluster(Z, criterion='distance', t=1.2)
	#biggest_cluster = get_biggest_cluster(clustering)

	files = os.listdir(input)

	if not os.path.isdir(output):
		os.mkdir(output)

	for i in range(len(clustering)):
		if not os.path.isdir(output + '/' + str(clustering[i])):
			os.mkdir(output + '/' + str(clustering[i]))
		
		print("Saving to " + output + '/' + str(clustering[i]) + '/' + str(files[i]))
		shutil.copyfile(input + '/' + files[i], output + '/' + str(clustering[i]) + '/' + files[i])
	
	#plot
	#if test:
	#	print("Biggest cluster: " + str(biggest_cluster))
	#	fig = plt.figure(figsize=(25, 10))
	#	dn = hcluster.dendrogram(Z)
	#	plt.show()
	
	return clustering
		
		
if __name__ == "__main__":
	test_random_floats()
	


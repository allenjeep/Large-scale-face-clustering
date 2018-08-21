
# coding: utf-8

# In[23]:


# rank order clustering
# based on the "zhu" paper
import numpy as np
import pandas as pd
import sys
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import os
import shutil
pic_size=4096
sample_size=100
neighbour_size= 10 # in paper it is set as 2 00

def Data_create(k):
	d= np.random.randint(0,10,size=(k,pic_size))
	return d
#test
# In[24]:

class Face:
	def __init__(self, 
				 neighbours = None, rank_list = None,  \
				 embedding = None, label = None, index =None):
		self.neighbours = neighbours
		self.rank_list = rank_list
		self.embedding = embedding
		self.index = index
		self.label = label # can be used for evaluating
#create faces list, also feature as its embedding, index 

def Face_creation(data):
	faces=[]
	for i in range(sample_size):
		face=Face()
		face.index = i
		face.embedding= data[i]
		faces.append(face)
	return(faces)


# In[25]:


class Cluster:
	def __init__(self):
		self.faces = list()
		self.neighbours = None
		self.rank_list = None
		self.normalized_distance = None


# In[26]:


class Neighbour: # neighbour can be face and cluster
	def __init__(self, entity, distance):
		self.entity = entity
		self.distance = distance # metric for sorting


# In[27]:


# build face neighbourhood based on the N1 distance
def neighbours_for_faces(faces, N = sample_size):  # decide for sample_size or neighbour size
	for i, face1 in enumerate(faces):
		neighbourhood = []
		for j, face2 in enumerate(faces):
			distance = np.linalg.norm(face1.embedding - face2.embedding, ord = 1)
			neighbour = Neighbour(face2, distance)
			neighbourhood.append(neighbour)
		neighbourhood.sort(key = lambda x: x.distance)
		face1.neighbours = neighbourhood[0:N]



# In[28]:


# entity1 and entiry2 is the vector index
def sym_rank_order(entity1, entity2):
	# order_e1_e2 can be interpreted as the number of rank order neighbour in entity 1
	distance_entity1_entity2, order_entity1_entity2 = asym_rank_order2(entity1, entity2)
	distance_entity2_entity1, order_entity2_entity1 = asym_rank_order2(entity2, entity1)
	min_neighbours = min(order_entity1_entity2, order_entity2_entity1)
	return((distance_entity1_entity2 + distance_entity2_entity1)/min_neighbours)

def asym_rank_order(entity1, entity2): # million
	distance = 0 # follow the sequence of entity1 list till find entity2 in the list
	for i in range(neighbour_size):
		penalty = 0
		for j in range(neighbour_size):
			if entity1.neighbours[i].entity is entity2.neighbours[j].entity:
				if j == 0: # utill it finds entity2 in its list
					return(distance, i+1)
				else:
					break
			else:
				penalty += 1
		if penalty == neighbour_size:
			distance +=1  # increase if not in the neighbour list
	return(distance, i+1)# confused of the min(oa,ob), if not found, set as neighbour size

def asym_rank_order2(entity1, entity2): #original rank_order
	distance = 0 # follow the sequence of entity1 list till find entity2 in the list
	for i, neighbour1 in enumerate(entity1.neighbours):
		for j, neighbour2 in enumerate(entity2.neighbours):
			if neighbour1.entity is neighbour2.entity:
				if j == 0: # utill it finds entity2 in its list
					return(distance, i+1)
				else:
					distance = distance+ j
	return (distance,i+1)

def rank_order_list(entities):
	for entity1 in entities:
		neighbourhood = []
		for entity2 in entities:
			if entity1 is entity2: # solve the problem of self distance
				neighbourhood.append(Neighbour(entity2, 0))
			else:
			# Get rank order distance between entity1 and face 2
				rank_order = sym_rank_order(entity1, entity2)
				neighbourhood.append(Neighbour(entity2, rank_order))
		neighbourhood.sort(key = lambda x : x.distance)
		entity1.rank_list = neighbourhood



# In[29]:


# Assigning each face to a cluster
def initial_cluster(faces):
	clusters = []
	for face in faces:
		cluster = Cluster() 
		cluster.faces.append(face)
		clusters.append(cluster)
	return(clusters)
#clusters= initial_cluster_creation(faces)


# In[30]:


# Cluster level order list
def cluster_distance(cluster1, cluster2): # traverse the face list in each cluster
	nearest_distance = sys.float_info.max
	for face1 in cluster1.faces:
		for face2 in cluster2.faces:
			distance = np.linalg.norm(face1.embedding - face2.embedding, ord = 1)
			if distance < nearest_distance: 
				nearest_distance = distance	
			# If there is a distance of 0 then there is no need to continue
			if distance == 0:
				return(0)
	return(nearest_distance)
			
# actually build the order list k is to determine top nearest, consider whole in each iteration		
def neighbours_for_clusters(clusters, K ):
	for i, cluster1 in enumerate(clusters):
		nearest_neighbours = []
		for j, cluster2 in enumerate(clusters):
			distance = cluster_distance(cluster1, cluster2)
			neighbour = Neighbour(cluster2, distance)
			nearest_neighbours.append(neighbour)
		nearest_neighbours.sort(key = lambda x: x.distance)
		cluster1.neighbours = nearest_neighbours[0:K]


# In[31]:


# cluster level normalized distance
def normalized_distance_clusters(cluster1, cluster2, K = 5):
	all_faces = cluster1.faces + cluster2.faces
	normalized_distance = 0
	for face in all_faces:
		total_distance = sum([neighbour.distance for neighbour in face.neighbours[0:K]]) 
		normalized_distance += total_distance
	K = min(len(face.neighbours), K) # in case order list is smaller than predifiend value
	normalized_distance = normalized_distance/K	
	# then divide by all the faces in the cluster
	normalized_distance = normalized_distance/len(all_faces)
	normalized_distance = (1/normalized_distance) * cluster_distance(cluster1, cluster2)
	return(normalized_distance)
# cluster level rank order distance
def rank_order_distance_clusters(cluster1, cluster2):
	return(sym_rank_order(cluster1, cluster2))


# In[32]:


# cluster 
def find_clusters(faces):
	clusters = initial_cluster(faces)
	neighbours_for_clusters(clusters,len(clusters))
	t = 10  # to be determined
	prev_cluster_number = len(clusters)
	num_created_clusters = prev_cluster_number
	is_initialized = True
	while (is_initialized) or (num_created_clusters): #first iteration or no more merge
		G = nx.Graph()
		for cluster in clusters:
			G.add_node(cluster)
		
		num_pairs = sum(range(len(clusters) + 1))
		processed_pairs = 0		
		for i, cluster1 in enumerate(clusters):
			for cluster_neighbour in cluster1.neighbours:
				cluster2 = cluster_neighbour.entity
				processed_pairs += 1
				if cluster1 is cluster2:
					continue
				else: 
					normalized_distance = normalized_distance_clusters(cluster1, cluster2)
					if (normalized_distance >= 1):  # condition 2
						continue
					rank_order_distance = rank_order_distance_clusters(cluster1, cluster2)
					if (rank_order_distance >= t):   # condition 1
						continue
					G.add_edge(cluster1, cluster2)  
		
		clusters = []
		for _clusters in nx.connected_components(G):
			new_cluster = Cluster()
			for cluster in _clusters:
				for face in cluster.faces:
					new_cluster.faces.append(face)
					
			clusters.append(new_cluster)
			
		current_cluster_number = len(clusters)
		num_created_clusters = prev_cluster_number - current_cluster_number
		prev_cluster_number = current_cluster_number		
		neighbours_for_clusters(clusters,len(clusters))
		is_initialized = False
		
	unmatched_clusters = []
	matched_clusters = []
	for cluster in clusters:
		if len(cluster.faces) == 1:
			unmatched_clusters.append(cluster)
		else:
			matched_clusters.append(cluster)	   
	matched_clusters.sort(key = lambda x: len(x.faces), reverse = True)		  
	return(matched_clusters, unmatched_clusters)

def save_cluster(cluster, cluster_output_directory, image_directory, clustering_type):
	print("Saving results for the " + clustering_type + " clustering algorithm")
	files = os.listdir(image_directory)

	if not os.path.isdir(cluster_output_directory):
		os.mkdir(cluster_output_directory)

	if not os.path.isdir(cluster_output_directory + '/' + clustering_type):
		os.mkdir(cluster_output_directory + '/' + clustering_type)

	for i in cluster:
		shutil.copyfile(image_directory + '/' + files[i],
											cluster_output_directory + '/' + clustering_type + '/' + files[i])

# In[33]:
def cluster(data, input, output):
	global pic_size, sample_size, neighbour_size
	files = os.listdir(input)
	
	if not os.path.isdir(output):
		os.mkdir(output)
	pic_size=len(data[0])
	sample_size=len(data)
	
	faces = Face_creation(data)
	
	neighbour_size= 10 # in paper it is set as 2 00
	neighbours_for_faces(faces)
	rank_order_list(faces)
	matched_clusters, unmatched_clusters = find_clusters(faces)
	
	for i,cl in enumerate(matched_clusters):
		if not os.path.isdir(output + '/' + str(i)):
			os.mkdir(output + '/' + str(i))
		
		for j, face in enumerate(cl.faces):
			print("Saving to: " + output + '/' + str(i) + '/' + files[face.index])
			shutil.copyfile(input + '/' + files[face.index],
							output + '/' + str(i) + '/' + files[face.index])

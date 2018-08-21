import os
import shutil
import sys
from libraries import vgg19
from libraries import rank_order
from libraries import cluster_mod
from libraries import normalize_faces

def save_cluster(biggest_cluster, cluster_output_directory, image_directory, clustering_type):
	print("Saving results for the " + clustering_type + " clustering algorithm")
	files = os.listdir(image_directory)
	
	if not os.path.isdir(cluster_output_directory):
		os.mkdir(cluster_output_directory)
		
	if not os.path.isdir(cluster_output_directory + '/' + clustering_type):
		os.mkdir(cluster_output_directory + '/' + clustering_type)	
		
	for i in biggest_cluster:
		shutil.copyfile(image_directory + '/' + files[i], 
						cluster_output_directory + '/' + clustering_type + '/' + files[i])

def main(image_directory, cluster_output_directory, weights_file, clustering_type):
	#if cluster_output_directory != "--skip":
	#	print("Finding faces")
	#	if not normalize_faces.normalize_faces(image_directory, image_directory + '/faces', delete=False):
	#		print("Error during face normalization. Exiting...")
	#		return
	
	print("Loading vgg19 model")
	(model, feature_layer) = vgg19.get_model(weights_file)
	
	print("Finding features...")
	features = vgg19.get_features(feature_layer, image_directory)
	
	print("Clustering...")
	if clustering_type in ['rank-order', 'all']:
		biggest_cluster = rank_order.cluster(features, image_directory, cluster_output_directory)#cluster_mod.cluster(features, image_directory, cluster_output_directory)
		#save_cluster(biggest_cluster, cluster_output_directory, image_directory + '/faces', 'rank-order1')
	
	if clustering_type in ['kmeans', 'all']:
		print("kmeans clustering not yet implemented")
	
	return

if __name__ == "__main__":
	if len(sys.argv) == 5:
		image_directory = sys.argv[1]
		cluster_output_directory = sys.argv[2]
		weights_file = sys.argv[3]
		clustering_type = sys.argv[4]
		
		print("Finding largest face cluster")
		print("image_directory: " + image_directory)
		print("cluster_output_directory: " + cluster_output_directory)
		print("weights_file: " + weights_file)
		print("clustering_type: " + clustering_type)
		
		if not clustering_type in ['rank-order', 'kmeans', 'all']:
			print("Invalid clustering type")
			print("Clustering methods:")
			print("rank-order, kmeans, all")
			exit()
		
		main(image_directory, cluster_output_directory, weights_file, clustering_type)
	else:
		print("Usage: train_net.py [image input directory] [cluster output directory] [weights file] [clustering method]")
		print("\nClustering methods:")
		print("rank-order, kmeans, all")

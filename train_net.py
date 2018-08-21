import sys
from libraries import extract_imdb
from libraries import normalize_faces
from libraries import vgg19
from libraries import rank_order

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


def main(imdb_metadata, imdb_image_directory, output_face_directory, input_weights_file, output_weights_file):
	if imdb_metadata == "--skip":
		print("Skipping face extraction")
	else:
		print("Extracting images using IMDB metadata")
		if not extract_imdb.generate_training_set(imdb_metadata, imdb_image_directory, output_face_directory + '/tmp'):
			print("Error during image extraction. Exiting...")
			return
			
		print("Normalizing faces...")
		if not normalize_faces.normalize_faces(output_face_directory + '/tmp', output_face_directory):
			print("Error during face normalization. Exiting...")
			return
	
	print("Training neural net")
	(m, lm) = vgg19.get_model(input_weights_file)
	vgg19.train_model(m, output_face_directory, epochs=5)
	
	#print("Model trained. Saving weights...")
	#vgg19.save_model(m, output_weights_file)
	
	print("Done!")

if __name__ == "__main__":
	if len(sys.argv) == 6:
		imdb_metadata = sys.argv[1]
		imdb_image_directory = sys.argv[2]
		output_face_directory = sys.argv[3]
		input_weights_file = sys.argv[4]
		output_weights_file = sys.argv[5]
		print("Training neural net for face clustering")
		print("imdbMeta: " + imdb_metadata)
		print("imdb_image_directory: " + imdb_image_directory)
		print("output_face_directory: " + output_face_directory)
		print("input_weights_file: " + input_weights_file)
		print("output_weights_file: " + output_weights_file)
		
		main(imdb_metadata, imdb_image_directory, output_face_directory, input_weights_file, output_weights_file)
	else:
		print("Usage: train_net.py [imdb metadata] [imdb image directory] [output face directory] [input weight file] [output weight file]")

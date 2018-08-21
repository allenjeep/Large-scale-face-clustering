import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.optimizers import SGD
import os
import cv2, numpy as np
import h5py

def print_layer(model, layers):
	for l in layers:
		print("layer " + l + " in: " + str(model.get_layer(name=l).input_shape))
		print("layer " + l + " out: " + str(model.get_layer(name=l).output_shape))

def VGG_19(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
	
	model.add(Conv2D(64, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3),  activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3),  activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3),  activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3),  activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu', name='features'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

def load_image(filename, flat=False):
	im = cv2.resize(cv2.imread(filename), (224, 224)).astype(np.float32)
	if flat:
		return np.expand_dims(im, axis=0)
	else:
		return im

def get_people(files):
	people = list()
	for f in files:
		name = f.split('.')[0]
		if not name in people:
			people.append(name)

	return people

def get_training_set(directory):
	files = [f for f in os.listdir(directory)]
	people = get_people(files)
	x_train = np.empty((len(files),224,224,3))
	y_train = np.empty((len(files),1000))
	
	print("Training on " + str(len(files)) + " images of " + str(len(people)) + " people")

	for i in range(len(files)):
		f = files[i]
		name = f.split('.')[0]
		x_train[i] = load_image(directory + '/' + f)
		y_train[i][people.index(name)] = 1.0
		
	return (x_train, y_train)

def get_model(weights_path='libraries/vgg19_weights_tf_dim_ordering_tf_kernels.h5'):
	# Test pretrained model
	model = VGG_19(weights_path)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')	
	layer_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
	return (model, layer_model)
	
def get_features(feature_layer, image_directory):
	files = os.listdir(image_directory)
	features = np.empty((len(files),4096))
	
	for i in range(len(files)):
		if i % 5 == 0:
			print("Extracting features... (" + str(i) + "/" + str(len(files)) + ")")
		f = files[i]
		image = load_image(image_directory + '/' + f, flat=True)
		features[i] = feature_layer.predict(image)
		
	print("Extracting features... (" + str(len(files)) + "/" + str(len(files)) + ")")
	print("Features extracted!")
	
	return features
	
def train_model(model, training_directory, batch_size=32, epochs=10):
	(x, y) = get_training_set(training_directory)
	model.fit(x, y, batch_size=batch_size, epochs=epochs)
	
def save_model(model, weight_file):
	model.save(weight_file)
	

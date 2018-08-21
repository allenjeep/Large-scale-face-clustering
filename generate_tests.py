import scipy.io
import numpy as np
import cv2
import shutil
import threading
import time
import os
import imutils
import random

imdbDirectory = '/data/Imdb-Wiki/imdb'
imdb = scipy.io.loadmat(imdbDirectory + '/imdb.mat')['imdb'][0][0]
testsets = '/data/Imdb-Wiki/imdb/testsets2'

# Folder structure
# testsets/
#	10/
#		originals/
#		faces/
#		clusters/
#			rank-order/
#				0/
#				1/
#			k-means/
#	20/
#	90/
rows = len(imdb[0][0])

colName = 4
colFile = 2
colCoord = 5
colFaceProb = 6
 
people = [
	'Jim Parsons',
	'Zooey Deschanel',
	'Neil Patrick Harris',
	'Nicole Kidman',
	'Robert Downey Jr.',
	'Tom Cruise',
	'Courteney Cox',
	'Angelina Jolie',
	'Brad Pitt']
	
count = np.zeros(len(people))

def ensure_dir(file_path):
    print("ensuring: " + file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_dir(name):
	ensure_dir(name + '/')
	ensure_dir(name + '/faces/')
	ensure_dir(name + '/original/')
	ensure_dir(name + '/clusters/')
	ensure_dir(name + '/clusters/rank-order/')

def process_image(directory, file, name, coordinates, row):
	if not os.path.isfile(imdbDirectory + '/' + file):
		print("File does not exist: " + imdbDirectory + '/' + file)
		return
	#try:
	shutil.copyfile(
		imdbDirectory + '/' + file, 
		directory + '/original/' + name + '.' + str(row) + '.jpg')
	
	image = cv2.imread(imdbDirectory + '/' + file)
	(x,y,w,h) = np.array(coordinates).astype(int)
	face = imutils.resize(image[y:h, x:w], width=244, height=244)
	
	cv2.imwrite(directory + '/faces/' + name + "." +  str(row) + ".jpg", face)
	print("Sucess: " + file + " (" + name + ")")
	#except:
	#	print("Fail  : " + str(row) + ":" + name)

ensure_dir(testsets)
for i in [10,20,30,40,50,60,70,80,90]:
	create_dir(testsets + '/' + str(i))
		
for row in range(2000):
	row = random.randint(0, rows - 1)
	if imdb[colName][0][row][0] in people:
		continue

	if str(imdb[colFaceProb][0][row]) == "-inf":
		continue

	while threading.active_count() > 8:
		time.sleep(1)
	
	threading.Thread(
		target=process_image,
		args=(
			testsets + '/random',
			imdb[colFile][0][row][0], 
			imdb[colName][0][row][0],
			imdb[colCoord][0][row][0].astype(int),
			row)
		).start()
		
		

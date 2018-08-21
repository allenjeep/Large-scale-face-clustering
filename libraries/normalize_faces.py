from libraries import face_aligner
from libraries import face_utils
import cv2
import sys
import os
import dlib
import imutils

def normalize_faces(input_dir, output_dir, delete=True):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("libraries/shape_predictor_68_face_landmarks.dat")
	fa = face_aligner.face_aligner(predictor)

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	

	counter = 0
	error = 0
	
	files = os.listdir(input_dir)

	for i in range(len(files)):
		f = files[i]
		try:
			if i % 5 == 0:
				print("Processing images.... " + str(i) + "/" + str(len(files)))
			
			# Read the image
			image = cv2.imread(input_dir + "/" + f)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			faces = detector(gray, 2)

			# Draw a rectangle around the faces
			for rect in faces:
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				faceOrig = imutils.resize(image[y:y + h, x:x + w], width=244)
				faceAligned = fa.align(image, gray, rect)
				
				cv2.imwrite(output_dir + '/' + f + '.' + str(counter) + '.jpg', faceAligned)
				counter += 1
		except:
			error += 1
		if delete:
			os.remove(input_dir + "/" + f)
			
	print("Processing images.... " + str(len(files)) + "/" + str(len(files)))
	print("Found " + str(counter) + " faces in " + str(len(files)) + " images!")
	
	if delete:
		os.rmdir(input_dir)
			
	return (counter > 0)

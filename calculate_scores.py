import os
import sys

people = ['Jim Parsons',
	'Zooey Deschanel',
	'Neil Patrick Harris',
	'Nicole Kidman',
	'Robert Downey Jr.',
	'Tom Cruise',
	'Courteney Cox',
	'Angelina Jolie',
	'Brad Pitt',
	'Jennifer Aniston']

def calculate_scores(directory):
	if not os.path.isdir(directory):
		print("No such directory: " + directory)
		return
	
	for i in range(1,10):
		subdir = directory + '/' + str(i * 10)
		#print("Opening: " + subdir)
		#print("Person of Interest: " + people[i])
		
		calculate_score(subdir + '/rank-order' , people[i])
		#files = os.listdir(directory)
		#for f in files:

def calculate_score(directory, person):
	files = os.listdir(directory)
	count = len(files)
	correct = 0
	
	for f in files:
		if person in f:
			correct += 1

	#print(directory + " " + str(correct) + " " + str(count) + " " + str(1.0*correct/count) + "")
	print(str(1.0*correct/count))

		

if __name__ == "__main__":
	if len(sys.argv) == 2:
		directory = sys.argv[1]
		print("Evaluating scores")
		print("directory: " + directory)
		calculate_scores(directory)
	else:
		print("usage: python calculate_scores.py [directory]")

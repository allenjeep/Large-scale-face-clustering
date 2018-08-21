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
		subdir = directory + '/' + str(i * 10) + '/clusters/rank-order1.2'
		score = ""
		for f in sorted(os.listdir(subdir)):
			score += calculate_score(subdir + '/' + f, people[i-1], i*10) + " " 
		print(str(i * 10) + ' ' + score)

def calculate_score(directory, person, cluster_size):
	files = os.listdir(directory)
	count = len(files)
	correct = 0
	
	for f in files:
		if person in f:
			correct += 1

	precision = 1.0*correct/count
	recall = 1.0*correct/cluster_size

	if precision+recall == 0:
		fmeasure = 0
	else:
		fmeasure = (2 * (precision*recall)/(precision+recall))

	return str(count) + ' ' + str(precision) + ' ' + str(recall) + ' ' + str(fmeasure)

	#return str(count) + ' ' + str(correct) 

		

if __name__ == "__main__":
	if len(sys.argv) == 2:
		directory = sys.argv[1]
		print("Evaluating scores")
		print("directory: " + directory)
		calculate_scores(directory)
	else:
		print("usage: python calculate_scores.py [directory]")

import pandas as pd
import numpy as np
import random
import math 
import time
import matplotlib.pyplot as plt	

text_file=open("docword.kos.txt","r")
lines = text_file.readlines()
text_file.close()

numberdocs = int(lines[0])
Vocab = int(lines[1])
numberwords = int(lines[2])

df = []

for i in range(3,len(lines)):
    df.append(lines[i].split(' '))

for i in range(len(df)):
    df[i][0] = int(df[i][0])
    df[i][1] = int(df[i][1])
    df[i][2] = int(df[i][2])


docs = {}   # every vocab for every doc is sorted in dictionary order and frequency is not accounted for
docfreq = [] 
docnorm = []
doclen = []

# docs dictionary starts from 1
# vocab starts from 1
# now every array is a np array 
#clusters starts from 0

for i in range(numberdocs + 1):
	docs[i] = []
	docnorm.append(0)

for i in range(Vocab + 1):
	docfreq.append(0)


start = time.time()

for i in range(numberwords):
	
	docs[df[i][0]].append([df[i][1], df[i][2]])
	docfreq[df[i][1]] += 1

for i in range(numberwords):
	w = (math.log10(numberdocs/docfreq[df[i][1]])*(df[i][2]))
	docnorm[df[i][0]] += w*w

for i in range(numberdocs +  1):
	docs[i] = np.array(docs[i])
	doclen.append(len(docs[i]))

docnorm = np.array(docnorm)
docnorm = np.sqrt(docnorm)


print(time.time()- start)
print("done")

def distance (doc1, doc2): # here we have two docs 
	
	intersect = 0.0 

	i = 0 
	j = 0
	n = len(doc1)
	m = len(doc2)

	while(i < n and j < m):
		
		if (doc1[i][0] == doc2[j][0]):
			intersect += 1
			i +=1 
			j += 1
		elif(doc1[i][0] < doc2[j][0]):
			i += 1
		else:
			j += 1

	r = (intersect)/(m + n - intersect)    # this is similarity

	return 1 - r

def assign_cluster(n, k, cluster, centroids): # is assign cluster to every point

	for i in range(k):
		cluster[i] = []

	for i in range(1, n + 1):
		
		assign = 0
		dist = 2
		
		a = [i for i in range(k)]

		a = np.random.permutation(np.array(a))
		a = a.tolist()

		for j in a:
			dist1 = distance(docs[i], centroids[j])
			if dist1 < dist:
				dist = dist1
				assign = j


		cluster[assign].append(i) 

	return

def assign_centroids(n, k, cluster, centroids):  # assigning centers to every cluster

	a = [i for i in range(k)]

	a = np.random.permutation(np.array(a))
	a = a.tolist()

	for i in a:

		centroids[i] = []
		mark = np.zeros((1,Vocab+1))
		m = len(cluster[i])

		for j in cluster[i]:
			doc = docs[j]
			for w in doc:
				
				mark[0][w[0]] += 1
				if(mark[0][w[0]] >= 0.10*m):
					mark[0][w[0]] *= -10
					centroids[i].append(w)
		
	return

def distance2(docid, cenid, centroidnorm, centroids2):

	dn = docnorm[docid]
	cn = centroidnorm[cenid]
	dot = 0

	doc = docs[docid]
	for w in doc:
		dot += (math.log10(numberdocs/docfreq[w[0]])*w[1])*centroids2[cenid][w[0]]

	if dn*cn == 0:
		return 1

	return 1 - dot/(dn*cn)

def assign_cluster2(n, k, cluster2, centroids2, centroidnorm): # is assign cluster to every point

	for i in range(k):
		cluster2[i] = []

	for i in range(1, n + 1):
		
		assign = 0
		dist = 2
		

		for j in range(k):
			dist1 = distance2(i, j, centroidnorm, centroids2)
			if dist1 < dist:
				dist = dist1
				assign = j

		cluster2[assign].append(i)

	return

def assign_centroids2(n, k, cluster2, centroids2, centroidnorm):  # assigning centers to every cluster

	centroids2 = np.zeros((k, Vocab + 1))

	for i in range(0, k):

		cs = len(cluster2[i])

		for d in cluster2[i]:
			for w in docs[d]:
				weight = math.log10(numberdocs/docfreq[w[0]])*w[1]
				centroids2[i][w[0]] += weight/cs
	
	centroidnorm = np.sqrt(np.sum(np.square(centroids2), axis = 1))
	
	return

def kmeans(n, k, itr):
	
	indx = random.sample(range(1,n + 1), k)
	centroids = {i : docs[indx[i]] for i in range(k)}
	
	cluster = {}

	centroids2 = np.zeros((k, Vocab + 1))

	for i in range(k):  #going through every centroid
		for w in docs[indx[i]]:# going through each vocab in each point in the centroid 
			centroids2[i][w[0]] = math.log10(numberdocs/docfreq[w[0]])*w[1]

	centroidnorm = {i: docnorm[indx[i]] for i in range(k)}

	
	cluster2 = {}

	while(itr > 0):

		itr -= 1

		assign_cluster(n, k, cluster, centroids)
		assign_centroids(n, k, cluster, centroids)

		assign_cluster2(n, k, cluster2, centroids2, centroidnorm)
		assign_centroids2(n, k, cluster2, centroids2, centroidnorm)

	error1 = 0
	error2 = 0
	error3 = 0
	error4 = 0
	for i in range(k):
		for j in cluster[i]:
			add = distance(docs[j], centroids[i])
			error1 += abs(add)
			error3 = max(error3, add)

	for i in range(k):
		for j in cluster2[i]:
			add = distance2(j, i, centroidnorm, centroids2)
			error4 = max(error4, add)
			error2 += abs(add)

	return [cluster,cluster2, centroids, error1, error2, error3, error4]


def main():


	error1 = []
	error2 = []
	error3 = []
	error4 = []

	maxk = 13

	for k in range(2,maxk):

		nc = k
		filename = "out.kos1." + str(nc) + ".txt"
		filename1 = "out.kos2." + str(nc) + ".txt"
		Ans = kmeans(numberdocs, nc, min(max(5,2*nc),25))

		cluster = Ans[0]
		cluster2 = Ans[1]
		centroids = Ans[2]
		error1.append(Ans[3]/100)
		error2.append(Ans[4]/100)
		error3.append(Ans[5])
		error4.append(Ans[6]*10)
		

		file = open(filename, 'w')

		for i in range(nc):
			file.write("The "+ str(i) + "th cluster is\n")
			file.write("number of docs :" + str(len(cluster[i])) + "\n")
			file.write("The centroid is : ")
			for j in centroids[i]:
				file.write(str(j[0])+ " ")
			file.write("\n" + str(cluster[i])+ "\n")

		file.close()
		file1 = open(filename1, 'w')

		for i in range(nc):
			
			file1.write("The "+ str(i) + "th cluster is\n")
			file1.write("number of docs :" + str(len(cluster2[i])))
			file1.write(str(cluster2[i])+ "\n")

		file1.close()

		print(time.time() - start)
		print(error1[k - 2])
		print(error2[k - 2])
		print(error3[k - 2])
		print(error4[k - 2])

		print("done")

	plt.axis([0,maxk,0,50])
	plt.plot([i for i in range(2,maxk)], error1, color ='r', label = 'Jaccard Measure(Cluster Error)')
	plt.plot([i for i in range(2,maxk)], error2, color ='g', label = 'td-idf Measure(Cluster Error)')
	plt.plot([i for i in range(2,maxk)], error3, color ='b', label = 'Jaccard Measure(Radius)')
	plt.plot([i for i in range(2,maxk)], error4, color ='y', label = 'td-idf Measure(Radius)')
	plt.xlabel('Number of Clusters')
	plt.ylabel('Total Error')
	plt.title("Error")
	plt.legend(loc='upper right')
	plt.show()


if __name__ == '__main__':
	main()


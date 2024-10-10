import sys
import numpy as np
import numpy.matlib as matlib
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
import random
import math
import time

CONSTANTE = -1 
np.set_printoptions(threshold=sys.maxsize) 

def parse(filename):
    try:
        X = np.genfromtxt(filename+'.txt', delimiter=' ')
        return X

    except IOError:
        print("abort..." , IOError.strerror)
        sys.exit(IOError.errno)



def centroids(X, U, k):
	global CONSTANTE
	d = X.shape[1] 
	M = np.zeros((d, k))
	for i in range (0,k):
		ind = np.argwhere(U[:,i]==1) 
		tem = X[ind] 
		temp = tem[:,0,:] 
		
		for j in range (0, d): 
			cpt=0 
			somme = 0
			for u in range(0,temp.shape[0]): 
				if(temp[u,j].any()!=CONSTANTE):
					somme += temp[u,j]
					cpt += 1
		
			
			
			M[j,i] = somme/cpt 



	return M

def angle(X, M, k):
	pass


def distance(X, M, k):
	N = len(X) 
	D = np.zeros((N, k)) 
	dim = len(X[0]) 
	for i in range(0,k):
		diff = X - matlib.repmat(M[:,i].T, N, 1) 
		D[:,i] = np.sum(np.power(diff,2), 1) 

	return D


def neoKMeans(X, k, alpha, beta, tmax, initU):
	C = [[]for i in range (0,k)] 
	N = len(X) 
	J = float('inf')
	oldJ = 0
	epsilon = 0 
	length = X.shape[0] 
	alphaN = math.floor(alpha*N)
	betaN = math.floor(beta*N)

	U = initU 
	t = 0 
	while(abs(oldJ-J)>epsilon and t < tmax):
		oldJ = J
		J = 0

		M = centroids(X, U, k)
		
		
		D = distance(X, M, k)
	

		
		T = []
		S = []
		p = 0
		C_c = set()  
		C_b = set()  
	
		

		nb_assign = N-betaN

		dist = np.min(D, axis=1)
		node = np.arange(0,N)
		ind = np.argmin(D, axis=1)
		dnk = np.zeros((N,3))
		dnk[:,0] = dist
		dnk[:,1] = node.T
		dnk[:,2] = ind

	
		
		dnk_sorted = dnk[dnk[:,0].argsort()] 
		sorted_d = dnk_sorted[:,0]
		sorted_n = dnk_sorted[:,1]
		sorted_k = dnk_sorted[:,2]

		
		J= J+ np.sum(sorted_d[1:nb_assign])

		
		temp = np.zeros((nb_assign, nb_assign))
	
		temp[:,0] = sorted_n[0:int(nb_assign),]
		temp[:,1] = sorted_k[0:int(nb_assign),]

		ind_no_cluster = dnk_sorted[nb_assign:N,1].astype('int')
		
		U[ind_no_cluster,] = 0

	
			
		for x in range (0,N-betaN):
			
			D[int(temp[x,0]), int(temp[x,1])] = float('inf') 
		
	

	

		n = 0
		while n < (alphaN + betaN):
			
			(i,j) = np.unravel_index(D.argmin(), D.shape)	
			min_d = np.min(D)
			J = J + min_d
			U[i,j] = 1
			D[i,j] = float('inf') 
			n = n+1
		
		t = t + 1

	return U



def display_cluster(X,U,k):
	length = X.shape[1]
		
	
	
	intersection = get_intersection(X,U)
	extreme = get_extreme(X,U)
 

 	
	c0 = X[np.argwhere(U[:,0]==1)]
	c1 = X[np.argwhere(U[:,1]==1)]
	if(k>2):
		c2 = X[np.argwhere(U[:,2]==1)]
	if(k>3):
		c3 = X[np.argwhereU[:,3]==1]

	fig, ax = plt.subplots(length,length, sharex='col', sharey='row')
	for i in range(0,length):
		for j in range(0, length):
			ax[i,j].set_xlim([-7,7])
			ax[i,j].set_ylim([-7,7])
			ax[i,j].plot(c0[:,0,i], c0[:,0,j],'ro')
			ax[i,j].plot(c1[:,0,i], c1[:,0,j],'bo')
			if(k>2):
				ax[i,j].plot(c2[:,0,i], c2[:,0,j],'mo')
			if(k>3):
				ax[i,j].plot(c3[:,0,i], c3[:,0,j],'yo')

			ax[i,j].plot(intersection[:,i], intersection[:,j],'go')
			ax[i,j].plot(extreme[:,i], extreme[:,j],'ko')
	plt.show()

	plt.pause(1)
	
	plt.close(fig)


def get_intersection(X,U):
	return X[np.where(np.all(U[:, :]==1, axis=1))[0]]

def get_extreme(X,U):
	return X[np.where(np.all(U[:,:]==0, axis=1))[0]]






path='dataset/tictactoefinal'
#path='dataset/zoofinal'
#path='dataset/votefinal'
#path='dataset/soybeanfinal'
#path='dataset/primaryTumorfinal'
#path='dataset/mushroomfinal'
#path='dataset/lymphfinal'

def parsev2(filename):
    data = []
    max_length = 0
    try:
        with open(filename + '.txt', 'r') as file:
            for line in file:
                
                row = list(map(int, line.strip().split()))
                if row[-1] == 0:
                    row = row[:-1] 
                data.append(row)
                if len(row) > max_length:
                    max_length = len(row)  

       
        data_normalized = [row + [-1]*(max_length - len(row)) for row in data]
        return np.array(data_normalized)

    except IOError as e:
        print("Error opening file:", str(e))
        sys.exit(e.errno)

#X = parsev2(path)
X = parse(path)
X = X[:, :-1]
print(X)




print("The number of transactions for a given data = ",len(X))




start_time = time.time()

k = 30
print("The number of fixed clusters k = ",k)
alpha = random.uniform(0,(k-1)/100) 
beta = random.uniform(0,1)
tmax = 100 


indX = km(k, random_state=0).fit_predict(X)


initU = np.zeros((len(X), k))
for j in range (0, k):
	initU[:,j] = indX==j




#alpha = 0.01
beta = 0.015

U = neoKMeans(X, k, alpha, beta, tmax, initU)


N = X.shape[0] 
nb_inter = len(U[np.where(U[:,0]==U[:,1])])




nb_inter = len(get_intersection(X,U))
nb_extrem = len(get_extreme(X,U))

i_Q1 = 0.25 
i_Q2 = 0.50 

e_Q1 = 0.001 
e_Q2 = 0.05 

b1 = (nb_inter > int(i_Q1*N)) and (nb_inter < int(i_Q2*N))
b2 = (nb_extrem > int(e_Q1*N)) and (nb_extrem < int(e_Q2*N))


oldinter=-1

b_inf = 0
b_sup = 1
end_time=time.time()
def jaccard_similarity(list1, list2):
    intersection_size = len(set(list1).intersection(set(list2)))
    union_size = len(set(list1).union(set(list2)))
    return intersection_size / union_size if union_size > 0 else 0

def intra_cluster_similarity(clusters):
    similarities = []

    for cluster in clusters:
       
        num_samples = len(cluster)
        similarity_matrix = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                similarity_matrix[i, j] = jaccard_similarity(cluster[i], cluster[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]

        sum_similarity = similarity_matrix.sum()
        similarities.append(sum_similarity)

    return similarities

def ICS(clusters):
    similarities = intra_cluster_similarity(clusters)
    s=0
    for i, similarity in enumerate(similarities):
        s+=similarity
    return s*0.5



def finalclusters(X, U, k):

    clusters = [[] for _ in range(k)]
    for i in range(len(X)):
        for j in range(k):
            if U[i, j] == 1:
                clusters[j].append([int(e) for e in list(X[i])])
    
    final_clus=[]
    for idx, cluster in enumerate(clusters):
        
        final_clus.append(list(cluster))
     
    return final_clus

print("Number of found clusters : ",len(finalclusters(X, U, k)))

tt=finalclusters(X, U, k)
qual=ICS(tt)
time=end_time-start_time
print("The quality ICS = ",qual)
print("time = ",time)



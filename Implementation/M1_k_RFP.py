import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary,LpStatus

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
               

    
def conceptual_clustering(transaction_data, patterns, beta_zero,teta,sigma):
   
    num_transactions = len(transaction_data)
    

    num_patterns = len(patterns)

  
    y = LpVariable.dicts("y", [j for j in range(num_patterns)], 0, 1, LpBinary)

 
    prob = LpProblem("ConceptualClustering", LpMaximize)

    # Objective function
    prob += lpSum(y[j] * len([T  for T in transaction_data if len(set(patterns[j])-set(T)) <=sigma ]) for j in range(num_patterns)), "Objective"

    # Constraint (1): Each transaction should be covered by exactly one pattern
    for i in range(num_transactions):
    
        prob += lpSum(1*y[j] for j in range(num_patterns) if len(set(patterns[j]) - set(transaction_data[i])) <= sigma ) == teta, f"Transaction_Coverage_{i}"


    # Constraint (2): Number of clusters should be exactly betazero
    prob += lpSum(y[j] for j in range(num_patterns)) == beta_zero, "Number_of_Clusters"

 
    prob.solve()
    soltime=prob.solutionTime
    
    print("the solution time :",soltime)

    clusters = []
    best_patterns=[]
    for j in range(num_patterns):
        if y[j].value() == 1:
            clusters.append([T  for T in transaction_data if len(set(patterns[j])-set(T)) <=sigma ])
            best_patterns.append(patterns[j])

 
    return (clusters,best_patterns,soltime)

    

if __name__ == "__main__":
    
    transaction_data =[]
    #path='dataset/tictactoefinal.txt'
    path='dataset/zoofinal.txt'
    #path='dataset/votefinal.txt'
    #path='dataset/soybeanfinal.txt'
    #path='dataset/primaryTumorfinal.txt'
    #path='dataset/mushroomfinal.txt'
    #path='dataset/lymphfinal.txt'
    with open(path,'r') as file:
        for line in file:
            et=line.split(" ")
            del et[-1]
            transaction_data.append([int (ee) for ee in et])
    print("number of transactions: ",len(transaction_data))
    print("from ",transaction_data[0], " ... to  ",transaction_data[len(transaction_data)-1])

    #provided patterns
    patterns=[]
    with open('RelaxedPatterns/patternsk1.txt', 'r') as file:
        for line in file:
            e=line.split(" ")
            if "\n" in e:
                e.remove("\n")
            if " " in e:
                e.remove(" ")
            if '' in e:
                e.remove('')
            patterns.append([int (ee) for ee in e])
    print("Relaxed Patterns generated :", len(patterns))

    cpu_res=[]
    ics_res=[]
    for beta_zero in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
        
        print("number of cluster chosen :", beta_zero)
    #cover allowance
        teta=1
    # relaxation of the pattern
        sigma=0

    # Solve the conceptual clustering problem
        result_clusters = conceptual_clustering(transaction_data, patterns, beta_zero,teta,sigma)

    # Print the result clusters
        print("number of founded clusters : ",len(result_clusters[0]))
       
        cpu_res.append((beta_zero,result_clusters[2]))
        ics_res.append((beta_zero,ICS(result_clusters[0])))
        print(result_clusters[1])
        
    print("CPU : ","\n")
    for e in cpu_res:
        print(e[0],"  ",e[1])
    print("ICS : ","\n")
    for e1 in ics_res:
        print(e1[0],"  ",e1[1])
        
   

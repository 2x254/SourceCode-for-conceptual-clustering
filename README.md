- This is the implementation of CCA-k-RFP-M1, CCA-k-RFP-M2, and other approaches. 

- The original dataset was collected from: https://dtai.cs.kuleuven.be/CP4IM/datasets/. It should be noted that we adapted the datasets to be suitable for the solver by increasing each item by 1 to remove the zero items, and also we added a zero value at each end of the line (transaction).

- To generate k-RFP, use the command below to run the solver. This is an example for generating classical patterns for the mushroom dataset:

--------> ./xsat4DAR -kx=0 -ky=1 -minsupp=1  -gdar=1 -msi=1 dataset/mushroomfinal.txt | grep "^[1-9]" >RelaxedPatterns/patternsk0.txt

- For the parameters, kx is relaxation, minsupp is the minimum support (alpha), and msi is the minimum item frequency (gamma).

- To generate k-RFP, you can choose a value of k >= 1, minsupp, gamma > 1, where (gamma=alpha for efficiency reasons).

- Install libraries using pip install ...

- Scripts: M1_Classical_Patterns.py (Classical closed patterns on M1), M1_k_RFP.py (relaxed patterns on M1), M2_Classical_Patterns.py (Classical closed patterns on M2), M2_k_RFP.py (relaxed patterns on M2), NeokMeans.py (overlapping k-Means), and Other_methods_VS_M1.py (other disjoint clustering methods for comparison with K-RFP applied on M1).
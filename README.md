ï»¿# DMML Assignment #2

#### Debjit Paria (BMC201704) & Ashwani Anand(BMC201605)

## About the program

We created a Python 2.7 program to compute K-means Clustering using Jaccard Distance and tf-idf distance on the Bag of Words dataset : [http://archive.ics.uci.edu/ml/datasets/Bag+of+Words](http://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

We read as input the bag of words and [output](https://github.com/debjit99/Clustering-text-documents/Output) the clusters(for the three datasets: KOS, Enron Emails & NIPS full papers) after _f(k)_ iterations with _k_ clusters, where $$f(k) = min \{2k, 25\}$$ We also output the following two graphs for the two distance measures: **Jaccard** & **tf-idf**:

- _the error_ vs _number of clusters graph_
-  _the max of the radius of all the clusters_ vs _number of clusters graph_
>**Error of a cluster:**  The sum distance of the documents in the cluster from the centroid of the same.
>**Radius of a cluster:** The distance between the centroid of the cluster and the farthest document from the centroid.


## Parameter
While calculating the centroid of the clusters using the Jaccard measure, we take into consideration only those words which are present in atleast P% of the documents in the respective clusters, where

- for KOS, P = 10
- for NIPS, P = 15
- for Enron, P = 8


## Libraries used

- numpy - to make the arrays
- random - to generate random seeds to initialize the k-means cluster
- math - to use mathematical functions like _log_
- time - to calculate the time taken for various operations in the program
- matplotlib.pyplot - To plot the analysis graphs mentioned above.

## Analysis of the output
The outputs and analysis for all the three data sets can be found [here](https://github.com/debjit99/Clustering-text-documents).
However for the sake of analysis here, we only considered the KOS file with $K \in [2, 12]$.

![Analysis Graph](https://raw.githubusercontent.com/debjit99/Clustering-text-documents/master/Figure_1.png)
For $K=10$, **tf-idf** gives better results (elbow structure), whereas for $K=11$, gives us a better result for **Jaccard measure**.

Radius for **tf-idf** is minimum, in these cases, for $K=10$ and that for **Jaccard measure** almost remains the same throughout.

Error in **Jaccard measure** remains stable, while for the **tf-idf**, the same fluctuates.

We conclude, from the observation of the clusters formed, that **tf-idf** has more stable cluster size than the **Jaccard measure**.


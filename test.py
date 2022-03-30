import numpy as np
from sklearn.cluster import KmeansBisecting
a = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [0, 2], [2, 1], [1, 2], [2, 2]])

est = KmeansBisecting(n_clusters=8, n_init=3);
#est2 = KMeans(); 
result = est.fit(a)
#result2 = est2.fit(a)
print(result)
import numpy as np
from sklearn.cluster import KmeansBisecting
a = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [0, 2], [2, 1], [1, 2], [2, 2]])


est = KmeansBisecting(n_clusters=8, n_init=3,random_state=3)

result = est.fit(a)
print(f'score: {result.score(a)}')

print(f'labels: {result.labels_} \ncenters:\n {result.cluster_centers_}')
print(f'inertia: {result.inertia_}')
print(f'score: {result.score(a)}')

b = np.reshape(np.asarray([np.random.normal(3,2) for x in range(100)]),(50,2))
est2 = KmeansBisecting(n_clusters=5, n_init=10,random_state=3)

result = est2.fit(b)

print(f'labels: {result.labels_} \ncenters:\n {result.cluster_centers_}')
print(len(result.labels_))
print(f'inertia: {result.inertia_}')
print(f'score: {result.score(b)}')
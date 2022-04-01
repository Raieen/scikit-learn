import numpy as np
from sklearn.cluster import KmeansBisecting
from scipy import sparse as sp
from sklearn.datasets import make_blobs

centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)
n_samples = 100
n_clusters, n_features = centers.shape
X, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)
X_csr = sp.csr_matrix(X)

def _check_fitted_model(km):
    # check that the number of clusters centers and distinct labels match
    # the expectation
    centers = km.cluster_centers_
    assert centers.shape == (n_clusters, n_features)

    labels = km.labels_
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert km.inertia_ > 0.0

# a = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [0, 2], [2, 1], [1, 2], [2, 2]])


# est = KmeansBisecting(n_clusters=8, n_init=3,random_state=3)

# result = est.fit(a)
# print(f'score: {result.score(a)}')

# print(f'labels: {result.labels_} \ncenters:\n {result.cluster_centers_}')
# print(f'inertia: {result.inertia_}')
# print(f'score: {result.score(a)}')

# b = np.reshape(np.asarray([np.random.normal(3,2) for x in range(100)]),(50,2))

# est2 = KmeansBisecting(n_clusters=5, n_init=10,random_state=3)

# result = est2.fit(b)

# print(f'labels: {result.labels_} \ncenters:\n {result.cluster_centers_}')
# print(len(result.labels_))
# print(f'inertia: {result.inertia_}')
# print(f'score: {result.score(b)}')

est3 = KmeansBisecting(init=np.asarray([[ 4.48411754e-03,  5.14092663e+00, 6.99763245e-03, -1.67330980e-01,
  -2.96512755e-01],
 [ 8.78951830e-01,  1.08375745e+00,  3.96596333e+00,  4.97245612e-02,
   1.75747563e-01],
 [ 8.74912533e-01,  4.56239522e-02, -2.78926596e-01,  5.36352151e+00,
   1.02411074e+00]]), n_clusters=3, random_state=42, n_init=1)

result = est3.fit(X_csr)

print(f'labels: {result.labels_} \ncenters:\n {result.cluster_centers_}')
print(len(result.labels_))
print(f'inertia: {result.inertia_}')
print(f'score: {result.score(X_csr)}')
print(result.cluster_centers_.shape)
print(n_clusters)
print(n_features)
_check_fitted_model(result)
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, MiniBatchKMeans


def cluster_vectors(docvs, method='DBS'):
    print("Calculating Clusters.")
    X = docvs
    if method == 'AP':
        af = AffinityPropagation(copy=False).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(set(labels))
    elif method == 'DBS':
        print("Computing DBSCAN")
        db = DBSCAN(eps=0.03, min_samples=5, algorithm='brute', metric='cosine').fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels))
    elif method == 'AC':
        print("Computing Agglomerative Clustering")
        ac = AgglomerativeClustering(10).fit(X)
        labels = ac.labels_
        n_clusters_ = ac.n_clusters
    elif method == 'KM':
        print("Computing MiniBatchKmeans clustering")
        km = MiniBatchKMeans(n_clusters=300, batch_size=200).fit(X)
        labels = km.labels_
        n_clusters_ = len(km.cluster_centers_)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    # % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    return X, labels

X, labels = cluster_vectors(docvs, "DBS")

def consulta(id,data):
    for i in data:
        if i['_id'] == id:
            return i
def extract_clustered_docs(ids, labels, cluster_label):
    docs = []
    for i, l in zip(ids, labels):
        if l !=cluster_label:
            continue
        d = consulta(i,data)
        docs.append(d["cleaned_text"])
    return docs

a = extract_clustered_docs(ids,labels,4)
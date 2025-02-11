def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5  #euclidean distance

def kmeans(data, k):
    centroids = data[:k]  #pick first k points as initial centroids
    clusters = [[] for _ in range(k)]

    for _ in range(5):  #fixed small number of iterations
        clusters = [[] for _ in range(k)]  #reset clusters
        for point in data:
            distances = [distance(point, centroid) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
        for i in range(k):
            if clusters[i]:  #avoid empty clusters
                x_sum, y_sum = 0, 0
                for x, y in clusters[i]:
                    x_sum += x
                    y_sum += y
                centroids[i] = (x_sum / len(clusters[i]), y_sum / len(clusters[i]))
    return clusters

#example usage
data = [(1, 1), (2, 2), (3, 3), (8, 8), (9, 9), (10, 10)]
print(kmeans(data, k=2))

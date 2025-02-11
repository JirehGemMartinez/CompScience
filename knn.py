def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5  # euclidean distance

def knn(train_data, train_labels, test_point, k):
    distances = [(distance(test_point, point), label) for point, label in zip(train_data, train_labels)]
    distances.sort()  #sort by distance
    nearest_labels = [label for _, label in distances[:k]]

    #find most common label
    return max(set(nearest_labels), key=nearest_labels.count)

#example usage
train_data = [(1, 2), (2, 3), (3, 4), (8, 8), (9, 9)]
train_labels = ['A', 'A', 'A', 'B', 'B']
test_point = (4, 5)
print(knn(train_data, train_labels, test_point, k=3))

import numpy as np

def dbi(data, labels):
    """
    :return: Ein Float-Wert, der die Clusterqualität misst (je niedriger, desto besser)
    """
    unique_labels = [label for label in np.unique(labels) if label != 3] #Noise ausschließen
    n_clusters = len(unique_labels)           # Anzahl der Cluster
    labels = np.array(labels)                 # Sicherstellen, dass labels ein NumPy-Array ist
    npData = np.array(data)

    # Zentroiden (Mittelpunkte) der Cluster berechnen
    centroids = []
    for label in unique_labels:
        cluster_points = npData[labels == label]
        centroids.append(np.mean(cluster_points, axis=0))
    centroids = np.array(centroids)

    # Durchschnittliche Intra-Cluster-Distanzen berechnen
    intra_distances = []
    for idx, label in enumerate(unique_labels):
        cluster_points = npData[labels == label]
        centroid = centroids[idx]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)  # Abstände zum Zentrum
        intra_distances.append(np.mean(dists))  # Durchschnittliche Intra-Distanz
    intra_distances = np.array(intra_distances)

    # Davies-Bouldin-Index berechnen: Maximaler Verhältniswert pro Cluster
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                inter_distance = np.linalg.norm(centroids[i] - centroids[j])
                ratio = (intra_distances[i] + intra_distances[j]) / inter_distance
                max_ratio = max(max_ratio, ratio)  # Maximalen Wert für Cluster i speichern
        db_index += max_ratio

    return db_index / n_clusters




def silhouette(data, labels):
    """
    :return: Array mit Silhouette-Werten (je näher an 1, desto besser der Punkt im Cluster)
    """
    npData = np.array(data)
    unique_labels = np.unique(labels)
    n_samples = len(npData)
    silhouette_vals = np.full(n_samples, np.nan)  # Standardwerte sind NaN (z. B. für Rauschen)

    # Noise auschließen
    for i, label in enumerate(labels):
        if label == 3:
            continue

        # Punkte im gleichen Cluster finden
        in_cluster = (labels == label)
        same_cluster = [npData[j] for j in range(len(labels)) if labels[j] == label and j != i]
        #same_cluster = npData[in_cluster]     #sorgt für nan

        # a(i): Durchschnittliche Distanz zu anderen Punkten im selben Cluster (ohne sich selbst)
        #a_i = np.mean(np.linalg.norm(same_cluster - npData[i],axis=1))
        a_i = np.mean([np.linalg.norm(point - npData[i])
                       for point in same_cluster if not np.array_equal(point, npData[i])])

        # b(i): Kleinste durchschnittliche Distanz zu einem anderen Cluster
        b_i = np.inf
        for other_label in unique_labels:
            if other_label != label and other_label != 3:
                other_points = npData[labels == other_label]
                center = np.mean(other_points, axis=0)
                dist = np.linalg.norm(center-npData[i])
                b_i = min(b_i,dist)

        # Silhouette-Wert berechnen: (b - a) / max(a, b)
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)

    return silhouette_vals  # Ein Array mit Silhouette-Werten (zwischen -1 und 1)


from collections import defaultdict
from itertools import permutations


from collections import defaultdict
from itertools import permutations

def calculate_best_label_combination(points, labels, pointsPerCluster):
    total_clusters = 3  # Expected true clusters

    # Generate correct labels (0, 1, 2 for clusters; 3 for noise)
    correct_labels = [
        i // pointsPerCluster if i < total_clusters * pointsPerCluster else 3
        for i in range(len(points))
    ]

    # Group indices by predicted label
    grouped_indices = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[label].append(index)

    # Sort label groups by label for consistency
    sorted_labels = sorted(grouped_indices.keys())
    index_groups = [grouped_indices[label] for label in sorted_labels]

    # Build per-cluster match ratios: one row per predicted cluster, 3 values for match against labels 0, 1, 2
    def match_ratios(group):
        size = len(group)
        return [
            sum(correct_labels[i] == target for i in group) / size
            for target in range(total_clusters)
        ]

    all_ratios = [match_ratios(group) for group in index_groups]

    # Fill up to 3 groups with zeroes if we have fewer than 3 predicted clusters
    while len(all_ratios) < total_clusters:
        all_ratios.append([0.0, 0.0, 0.0])

    # Try all permutations of labels 0–2 mapped to index groups 0–2
    best_sum = -1
    best_perm = None

    for perm in permutations(range(total_clusters)):
        current_sum = sum(all_ratios[i][perm[i]] for i in range(total_clusters)) / total_clusters
        if current_sum > best_sum:
            best_sum = current_sum
            best_perm = perm

    return best_sum



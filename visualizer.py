import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def drawCAResult(data, membershipList, noiseGroup=False, dvv=False):
    clusterCount = max(membershipList) + 1
    clusters = [[] for i in range(clusterCount)]
    for p in range(len(data)):
        clusters[membershipList[p]].append(data[p])

    colors = ['blue', 'red', 'pink', 'black', 'orange', 'green']

    for i in range(clusterCount):
        xVals = [p[0] for p in clusters[i]]
        yVals = [p[1] for p in clusters[i]]
        if i == clusterCount - 1 and noiseGroup:
            plt.scatter(xVals, yVals, color=colors[i], label=f'Noisecluster', s=5)
        else:
            plt.scatter(xVals, yVals, color=colors[i], label=f'Cluster {i+1}', s=5)

    if dvv:
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
    else:
        plt.xlim(0, 1,)
        plt.ylim(0, 1)
    plt.legend()


def drawCluster(data, k, points, dvv=False):
    clusters = [[] for i in range(k+1)]
    counter = 0
    for cluster in range(k):
        for point in range(points):
            clusters[cluster].append(data[counter])
            counter += 1

    for i in range(len(data)):
        if i>=counter:
            clusters[k].append(data[i])

    colors = ['blue', 'red', 'pink', 'black', 'orange', 'green']

    for i in range(k+1):
        xVals = [p[0] for p in clusters[i]]
        yVals = [p[1] for p in clusters[i]]
        plt.scatter(xVals, yVals, color=colors[i], label=f'Cluster {i + 1}', s=5)

    if dvv:
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)

    else:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.legend()
    plt.show()


def drawMultipleCAResults(dataArray, membershipListArray, hasNoiseArray=[], dvv=False):
    for i in range(len(dataArray)):
        plt.figure()
        drawCAResult(dataArray[i], membershipListArray[i], hasNoiseArray[i] if (len(hasNoiseArray) > 0) else False, dvv)

    plt.show()


def drawSilhouttePlot(scoreData,membershipList, noiseCluster=False):
    clusteredData = [[] for i in range(max(membershipList) + 1)]
    Labels = []

    for p in range(len(scoreData)):
        clusteredData[membershipList[p]].append(scoreData[p])

    for array in range(len(clusteredData)):
        clusteredData[array] = sorted(clusteredData[array], reverse=True)


    orderedData = []
    counter = 0
    for list in range(len(clusteredData)):
        for s in range(len(clusteredData[list])):
            orderedData.append(clusteredData[list][s])
            Labels.append(counter)
            counter += 1
        if list < len(clusteredData)-1:
            for i in range(round(len(scoreData)/10)):
                orderedData.append(0)
                Labels.append(counter)
                counter += 1

    plt.barh(Labels, orderedData)
    plt.ylabel('Points')
    plt.xlabel('Score')
    plt.gca().invert_yaxis()
    plt.show()


def drawIndexGraph(scoreData, indexLabel='Silhouette-Score', iterationLabel='k'):
    plt.xlabel(iterationLabel)
    plt.ylabel(indexLabel)

    xVals = list(range(1, 1 + len(scoreData)))
    plt.plot(xVals, scoreData, marker='o')

    plt.show()


def drawTable(all_statistics, title="Clustering Results"):
    headers = ["Algorithm", "Dataset", "DVV", "Avg Percentages", "Avg SC", "Avg DBI"]
    algorithm_names = {0: "K-Means", 1: "K-Me. N", 2: "DBSCAN"}  # Zuordnung der Algorithmenamen
    table = []

    for index, stats in enumerate(all_statistics):
        algo = index // 8  # Es gibt 3 Algorithmen (0, 1, 2)
        algorithm_name = algorithm_names.get(algo, f"Algo {algo}")
        dataset = (index) % 4  # 4 Datensatztypen (d1, d2, d3, d4)
        dvv = 0 if index%8 < 4 else 1  # 2 DVV-Optionen (False, True)

        avg_percentages = stats[0]
        avg_silhouette = stats[1]
        avg_dbi = stats[2]

        table.append([
            algorithm_name,
            f"d{dataset + 1}",
            "On" if dvv else "Off",
            round(avg_percentages[0], 2),
            round(avg_silhouette[0], 2),
            round(avg_dbi[0], 2)
        ])

    # Dynamische Anpassung der Abbildungsgröße
    num_rows = len(table)  # Anzahl der Zeilen in der Tabelle
    num_cols = len(headers)  # Anzahl der Spalten
    fig_height = max(10, num_rows * 0.5)  # Dynamischer Plot-Höhenwert, mindestens 6
    fig_width = max(10, num_cols * 0.5)  # Dynamischer Plot-Breitenwert, mindestens 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")

    tbl = ax.table(
        cellText=table,  # Tabelleninhalte
        colLabels=headers,  # Spaltenüberschriften
        cellLoc="center",  # Zentrierten Text
        loc="center"  # In der Mitte positionieren
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(16)
    tbl.scale(1, 2)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    # Titel hinzufügen
    plt.title(title, fontsize=14, pad=20)

    plt.show()

def visualize_data(value_array, title="Data Visualization", xlabel="Index", ylabel="Value", color="blue", label="Data"):
    def flatten(data):
        if isinstance(data, (list, tuple, np.ndarray)):
            return [item for sublist in data for item in flatten(sublist)]
        else:
            return [data]

    flat_values = flatten(value_array)

    x_values = np.arange(1, len(flat_values) + 1)

    plt.scatter(x_values, flat_values, marker='o', color=color, label=label)

    # Diagrammeinstellungen
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if len(x_values) > 33:
        plt.xticks(
            np.linspace(1, len(flat_values), num=min(len(flat_values), 50), dtype=int)
        )
    else:
        plt.xticks(x_values)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()



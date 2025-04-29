import matplotlib.pyplot as plt
import matplotlib.cm as cm


def drawCAResult(data, membershipList, noiseGroup=False):
    clusterCount = max(membershipList) + 1
    clusters = [[] for i in range(clusterCount)]
    for p in range(len(data)):
        clusters[membershipList[p]].append(data[p])

    colors = ['blue', 'red', 'green', 'black', 'orange', 'pink']

    for i in range(clusterCount):
        xVals = [p[0] for p in clusters[i]]
        yVals = [p[1] for p in clusters[i]]
        if i == clusterCount - 1 and noiseGroup:
            plt.scatter(xVals, yVals, color=colors[i], label=f'Noisecluster', s=5)
        else:
            plt.scatter(xVals, yVals, color=colors[i], label=f'Cluster {i+1}', s=5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()


def drawMultipleCAResults(dataArray, membershipListArray, hasNoiseArray=[]):
    for i in range(len(dataArray)):
        plt.figure()
        drawCAResult(dataArray[i], membershipListArray[i], hasNoiseArray[i] if (len(hasNoiseArray) > 0) else False)

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

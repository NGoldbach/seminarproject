import matplotlib.pyplot as plt
import matplotlib.cm as cm


def drawCAResult(data, membershipList, noiseGroup=False):
    clusterCount = max(membershipList) + 1
    clusters = [[] for i in range(clusterCount)]
    for p in range(len(data)):
        clusters[membershipList[p]].append(data[p])

    colors = cm.get_cmap('tab10', clusterCount)

    for i in range(clusterCount):
        xVals = [p[0] for p in clusters[i]]
        yVals = [p[1] for p in clusters[i]]
        if i == clusterCount - 1 and noiseGroup:
            plt.scatter(xVals, yVals, color=colors(i), label=f'Noisecluster')
        else:
            plt.scatter(xVals, yVals, color=colors(i), label=f'Cluster {i}')

    plt.legend()


def drawMultipleCAResults(dataArray, membershipListArray, hasNoiseArray=[]):
    for i in range(len(dataArray)):
        plt.figure()
        drawCAResult(dataArray[i], membershipListArray[i], hasNoiseArray[i] if (len(hasNoiseArray) > 0) else False)

    plt.show()


def drawSilhouttePlot():
    pass


def drawDBIGraph():
    pass


def drawAverageSilhoutteScoreGraph():
    pass

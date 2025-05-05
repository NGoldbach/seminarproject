import data_preprocessor as dp
import evaluator as ev
import main_processor as mp
import visualizer as vs
import data_generator as dg
import numpy as np

def testrun(dn, algo, dvv, pointsperCluster):
    avgPercentages = []
    avgTotalSc = []
    avgDbi = []
    percentageArrays = []
    scArrays = []
    dbiArrays = []
    for i in range(len(dn)):
        currentDataset = dn[i]
        if dvv:
            currentDataset = dp.standardize(currentDataset)
        result = None
        if algo == 0:
            print("kmeans")
            result = mp.kmeans(currentDataset,3,False)
            result = result[0]
        elif algo == 1:
            print("kmeans n")
            result = mp.kmeans(currentDataset,3,True) #sclar noch anpassen!
            result = result[0]
        else:
            print("dbscan")
            result = mp.dbscan(currentDataset, 0.4 if not dvv else 0.2, 4)#eps x if dvv else eps y

        correctCounter = 0
        for j in range(3):
            for k in range(pointsperCluster):
                if j == 0 and result[j*pointsperCluster + k] == j:
                    correctCounter += 1
                if j == 1 and result[j*pointsperCluster + k] == j:
                    correctCounter += 1
                if j == 2 and result[j*pointsperCluster + k] == j:
                    correctCounter += 1

        for j in range(pointsperCluster * 3, pointsperCluster * 3 +(len(result)- 3 * pointsperCluster)): #check for correct indizes, 300 to max
            if result[j] == 3:
                correctCounter += 1
        percentageCorrect = correctCounter/len(result)
        percentageArrays.append(percentageCorrect)

        if len(set(result)) > 1:
            sc = ev.silhouette(currentDataset, result)
            scArrays.append(np.nanmean(sc))
            dbi = ev.dbi(currentDataset, result) # <- fix for noise
            dbiArrays.append(dbi)
        else:
            scArrays.append(-1)
            dbiArrays.append(10) # figure out sensible value

    avgPercentages.append(np.mean(percentageArrays))
    avgTotalSc.append(np.mean(scArrays))
    avgDbi.append(np.mean(dbiArrays))

    return [avgPercentages,avgTotalSc,avgDbi,percentageArrays,scArrays,dbiArrays]
#test noise function, works
initialData = dg.createDataSet(100,3)
vs.drawCluster(initialData[0],3,100)
vs.drawCluster(dp.standardize(initialData[0]),3,100, True)
for i in range(4):
    noisedData = dg.noiseGenerator(initialData,i)
    vs.drawCluster(noisedData,3,100)
    vs.drawCluster(dp.standardize(noisedData),3,100, True)
exit()

#pre set-up: create all datasets for tests and variants: low/high noise, far/near noise -> 4 dataset types, d1,d2,d3,d4

dnSize = 20 #anzahl der datensets d für die arrays d1,d2,..
pointsPerCluster = 20
clusterCount = 3
dg.clearDataSetFile()

for i in range(4):
    for j in range(dnSize):
        newData= dg.createDataSet(pointsPerCluster,clusterCount)
        noisedData = dg.noiseGenerator(newData, i) #i wird verwendet, um den modus der noise generation zu wählen
        dg.saveDataSet(noisedData)

#setup data
dn = [[],[],[],[]] #low/far, high/far, low/near, high/near
data = dg.getDataSets()
counter = 0

for i in range(4):
    for j in range(int(len(data)/4)):
        #separat, damit wir nicht immer daten erzeugen müssen, sobald wir mit obigen code fertig sind
        dn[i].append(data[counter])
        counter += 1

#bis hier alles funktional und getestet

#for all 24 variants(3 algos, d1-d4, dvv on/off)
allStatistics = [] #gets the 24 variants of [avgPercentages,avgTotalSc,avgDbi,percentageArrays,scArrays,dbiArrays]
for algo in range(3): #0 -> kmeans, 1-> kmeans noise, 2->dbscan
    for dvv in range(2): #0 -> False, != 0 -> True in python
        for dataset in range(4): #d1,d2,d3,d4
            currentStats = testrun(dn[dataset],algo,dvv,pointsPerCluster)
            allStatistics.append(currentStats)
            #could create table/visual here already, whatever is easier for you


# Test-Ausgabe 
#print(allStatistics)

#tabelle = avg pc, avg total sc, avg total dbi
# Tabelle mit den Werten: avgPercentages, avg total silhouette score, avg total Davies-Bouldin-Index
vs.drawTable(allStatistics)


#visualization = [pc1,pc2,pc3], [avg. sc1,avg. sc2,avg. sc3,..], [dbi1,dbi2,dbi3]

# visualizaion für PC
vs.visualize_data(
    value_array=[x[0] for x in allStatistics],
    title="Average Percentages",
    xlabel="Configuration",
    ylabel="Avg Percentage",
    color="orange",
    label="Avg Percentages"
)

# visualizaion für SC
vs.visualize_data(
    value_array=[x[1] for x in allStatistics],
    title="Average Silhouette Scores",
    xlabel="Configuration",
    ylabel="Silhouette Score",
    color="green",
    label="Silhouette Scores"
)

# visualization für DBI
vs.visualize_data(
    value_array=[x[2] for x in allStatistics],  # Extrahiere die Werte an Index 2
    title="Average Davies-Bouldin Index",
    xlabel="Configuration",
    ylabel="DBI",
    color="blue",
    label="DBI Values"
)








#kmeans test
# testData = [(0.3,0.3), (0.401, 0.398), (0.399, 0.402), (0.400, 0.401), (0.403, 0.400), (0.398, 0.399), (0.402, 0.397), (0.400, 0.400), (0.401, 0.403), (0.397, 0.401), (0.400, 0.399), (0.6, 0.6), (0.633, 0.633), (0.667, 0.667), (0.7, 0.7), (0.733, 0.733), (0.767, 0.767), (0.8, 0.8), (0.833, 0.833), (0.867, 0.867), (0.9, 0.9)]
# testData = dp.standardize(testData)
# currentScore = -2
# savedLabels = []
# for i in range(100):
#     print(i)
#     currentResult = mp.kmeans(testData, 2,True)
#     tempScores = ev.silhouette(testData, np.array(currentResult[0]))
#     avgScore = np.mean(tempScores)
#     if avgScore > currentScore:
#         print("now")
#         print(currentScore, avgScore)
#         currentScore = avgScore
#         savedLabels = currentResult
# vs.drawMultipleCAResults([testData],[currentResult[0]],[True])

# data = [
#     -20, 5, 30,
#     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
#     10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11
# ]
#
# stData = dp.standardize(data)
# print(stData)

# for i in range(1):
#     dg.createDataSet(100,2,1,3)
#     preprocessData = dg.getDataSets()
#     preprocessData = preprocessData[-1]
#     resultPre = None
#     currentScore = -2
#     for i in range(10):
#         currentResult = mp.kmeans(preprocessData, 2,True)
#         tempScores = ev.silhouette(preprocessData, np.array(currentResult[0]))
#         avgScore = np.mean(tempScores)
#         if avgScore > currentScore:
#             currentScore = avgScore
#             resultPre = currentResult[0]
#     print("Resultpre: ", resultPre)
#     vs.drawMultipleCAResults([preprocessData],[resultPre],hasNoiseArray=[True])
#     postprocessData = dp.standardize(preprocessData)
#     resultPost = None
#     currentScore = -2
#     for i in range(10):
#         currentResult = mp.kmeans(postprocessData, 2,True)
#         tempScores = ev.silhouette(postprocessData, np.array(currentResult[0]))
#         avgScore = np.mean(tempScores)
#         if avgScore > currentScore:
#             currentScore = avgScore
#             resultPost = currentResult[0]
#     print("Resultpost: ", resultPost)
#     vs.drawMultipleCAResults([postprocessData],[resultPost],hasNoiseArray=[True])


# dbscan test
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# testDBSCAN = mp.dbscan(datasets[-1], 0.05, 4)
# vs.drawCAResult(datasets[-1], testDBSCAN, False)

#dbi test
#dbiData = [12,52,5,9,100,22,75]
#vs.drawDBIGraph(dbiData)

#silouhette plot test
#sptData = [.5,-.4,.1,.2,.3,.9,-.8,.6,.5,-.4,.1,.8,-.9,-.2]
#sptMemb = [2,1,0,0,0,2,1,1,1,1,1,2,2,2]
#vs.drawSilhouttePlot(sptData, sptMemb)

#generator test
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# noNoiseTest = mp.kmeans(datasets[-1],[datasets[-1][11],datasets[-1][111],datasets[-1][211]])
# vs.drawMultipleCAResults([datasets[-1]],[noNoiseTest[0]],[False])

# DBSCAN test II
# dg.createDataSet(100,3,1,1)
# datasets = dg.getDataSets()
# data = datasets[-1]
# labels = mp.dbscan(data)
# vs.drawMultipleCAResults([data], [labels], [False])
# print(labels)

#metriken test
# dg.createDataSet(100,3,1,1)
# testData = dg.getDataSets()[-1]
#
# kmeans_result = mp.kmeans(testData,3)
# kmeans_labels = np.array(kmeans_result[0])
# dbi_kmeans = ev.dbi(testData, kmeans_labels)
# print("Kmeans DBI: ", dbi_kmeans, "\n")
# silhouette_kmeans = ev.silhouette(testData, kmeans_labels)
# print("Kmeans Silhouette: ", silhouette_kmeans, "\n")
# vs.drawIndexGraph([dbi_kmeans])
# vs.drawSilhouttePlot(silhouette_kmeans, kmeans_labels)
# vs.drawMultipleCAResults([testData],[kmeans_labels],[False])
#
# # DBSCAN Test
# dbscan_labels = np.array(mp.dbscan(testData))
# print("DBSCAN -Cluster Labels: ", dbscan_labels, "\n")
# silhouette_dbscan = ev.silhouette(testData, dbscan_labels)
# print("DBSCAN Silhouette: ", silhouette_dbscan, "\n")
# dbi_dbscan = ev.dbi(testData, dbscan_labels)
# print("DBSCAN -DBI: ", dbi_dbscan, "\n")
# vs.drawIndexGraph([dbi_dbscan])
# vs.drawSilhouttePlot(silhouette_dbscan, dbscan_labels)
# vs.drawMultipleCAResults([testData],[dbscan_labels],[False])
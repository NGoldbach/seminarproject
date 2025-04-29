import data_preprocessor as dp
import evaluator as ev
import main_processor as mp
import visualizer as vs
import data_generator as dg
import numpy as np

#kmeans test
# testData = [(0.1,0.1),(0.2,0.2),(0.5,0.5),(5,5),(6,6),(5.5,5.5),(1,19)]
# noNoiseTest = mp.kmeans(testData,2,nvBool=False)
# withNoiseTest = mp.kmeans(testData,2,nvBool=True)
# vs.drawMultipleCAResults([testData,testData],[noNoiseTest[0],withNoiseTest[0]],[False,True])

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
dg.createDataSet(100,3,1,1)
testData = dg.getDataSets()[-1]

kmeans_result = mp.kmeans(testData,3)
kmeans_labels = np.array(kmeans_result[0])  # Konvertiere Labels in NumPy-Array
dbi_kmeans = ev.dbi(testData, kmeans_labels)
print("Kmeans DBI: ", dbi_kmeans, "\n")
silhouette_kmeans = ev.silhouette(testData, kmeans_labels)
print("Kmeans Silhouette: ", silhouette_kmeans, "\n")
vs.drawIndexGraph([dbi_kmeans])
vs.drawSilhouttePlot(silhouette_kmeans, kmeans_labels)
vs.drawMultipleCAResults([testData],[kmeans_labels],[False])

# DBSCAN Test
dbscan_labels = np.array(mp.dbscan(testData))  # Ersetze 'data' durch 'testData'
print("DBSCAN -Cluster Labels: ", dbscan_labels, "\n")
silhouette_dbscan = ev.silhouette(testData, dbscan_labels)  # Ersetze 'data' durch 'testData'
print("DBSCAN Silhouette: ", silhouette_dbscan, "\n")
dbi_dbscan = ev.dbi(testData, dbscan_labels)  # Ersetze 'data' durch 'testData'
print("DBSCAN -DBI: ", dbi_dbscan, "\n")
vs.drawIndexGraph([dbi_dbscan])
vs.drawSilhouttePlot(silhouette_dbscan, dbscan_labels)
vs.drawMultipleCAResults([testData],[dbscan_labels],[False])
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import matplotlib.cm
cmap=matplotlib.cm
cmap=matplotlib.cm.get_cmap('plasma')
from sklearn.cluster import KMeans

data = pd.read_csv('Mall_Customers.csv')
X=data.iloc[:,[3,4]]
X.head()

wcss=[]

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)

wcss

plt.figure()
plt.plot(range(1,21), wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
y_kmeans=kmeans.fit_predict(X)
y_kmeans

Group_cluster=pd.DataFrame(y_kmeans)
Group_cluster.columns=['Group']
full_data=pd.concat([data, Group_cluster], axis=1)
print(full_data)

kmeans_pred = KMeans(n_clusters=k, random_state=42).fit(X)
kmeans_pred.cluster_centers_

kmeans_pred.predict([[100,50],[30,80]])

labels = [('Cluster'+str(i+1)) for i in range(k)]
print(labels)

X=np.array(X)
plt.figure()
for i in range(k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=20, c=cmap(i/k), label=labels[i])


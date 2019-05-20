#%%all
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/transaction.csv')

grouped_country_df = df.groupby('Country')

list_count = list()
for key, item in grouped_country_df:
    list_count.append(
        [1, len(item.groupby('InvoiceNo')),
         np.unique(item['Country'])])

list_count = np.array(list_count)

kmeans = KMeans(n_clusters=3)

new_cluster = kmeans.fit_predict(list_count[:, :2])

new_data = pd.DataFrame({
    'x': list_count[:, 0],
    'y': list_count[:, 1],
    'country': list_count[:, 2],
    'cluster': new_cluster[:],
})

plt.figure('Transaction Data Clustering with K-Means')
plt.scatter(new_data['x'].values,
            new_data['y'].values,
            s=100,
            c=new_data['cluster'].values)
plt.yticks(new_data['y'].values, new_data['country'])
plt.show()

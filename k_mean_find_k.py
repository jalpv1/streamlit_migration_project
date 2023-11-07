import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
# color = sns.color_palette()
# style.use('seaborn-white')
# style.available
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']

df = pd.read_csv("americabirthplace.csv", sep=",")
print(df.head())
df.drop(index = df.loc[df['AREA']==999].index.tolist(), inplace=True)
for year in years:
    df[year] = pd.to_numeric(df[year], errors='coerce')
    m = np.mean(df[year])
    df[year] = df[year].replace(np.nan,m)

df_n2 = df[years]
scaler = StandardScaler()
# Apply the fit transform method in a variable called df_std
df_std = scaler.fit_transform(df_n2)
# Looking at the standardized array
print(df_std)
wcss = []
# run a for loop for trying out several solutions (I’ll go for 10 times) and add the values on that list
for i in range(1,11):
    # "k-means++" is an algorithm that runs before the actual K-Means to find the best starting points for the centroids
    kmeans = KMeans(n_clusters=i, init="random", random_state=42)
    # fit the k-means model using the standardized data
    kmeans.fit(df_n2)
    # store the value inside wcss in the inertia_ attribute
    wcss.append(kmeans.inertia_)
plt.figure(figsize = (10,8))
plt.plot(range(1,11), wcss, marker = "o", linestyle = "--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("K-Means Clustering", fontsize=10, loc='right')
plt.show()
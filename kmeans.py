import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']

df = pd.read_csv("americabirthplace.csv", sep=",")
#print(df.head())
df.drop(index = df.loc[df['AREA']==999].index.tolist(), inplace=True)
for year in years:
    df[year] = pd.to_numeric(df[year], errors='coerce')
    m = np.round(np.mean(df[year]))
    df[year] = df[year].replace(np.nan,m)
    df[year] = df[year].replace(0,m)

#df.drop(["Type"], axis=1, inplace=True)
df_n2 = df[years]
#df_n2 = df_n.replace(np.nan, 0)
df_n2 = df[years]
scaler = StandardScaler()
df_std = scaler.fit_transform(df_n2)

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=0)
kmeans.fit(df_std)
df_km = df_n2.copy()
df_km2 = df_km.copy()
df_km["Segment K-means"] = kmeans.labels_
#df_km["Country"] = df["OdName"]
gr = df_km[df_km['Segment K-means'] == 0]
df_km_analisys = df_km.groupby(["Segment K-means"]).count()

print(df_km_analisys)

df_km2["Country"] = df["OdName"]
df_km2["Segment K-means"] = kmeans.labels_
#
# map = {}
# for cluster in [0,1,2,3,4,5]:
#     column = df_km2[df_km2["Segment K-means"]==cluster]
first_df =df_km2[df_km2["Segment K-means"]==0][: 30]
import plotly.graph_objects as go

df_colums = list(["Country_Name", "1980", "1990", "2000", "2005","2013"])
fig = go.Figure(data=[go.Table(
    header=dict(values=df_colums,
                fill_color='lightblue',
                align='left'),
    cells=dict(values=[first_df["Country"],first_df["1980"],first_df["1990"],first_df["2000"],first_df["2005"],first_df["2013"]],
               fill_color='lavender',
               align='left'))
])
fig.show()

second_df =df_km2[df_km2["Segment K-means"]==1]
fig = go.Figure(data=[go.Table(
    header=dict(values=df_colums,
                fill_color='lightblue',
                align='left'),
    cells=dict(values=[second_df["Country"],second_df["1980"],second_df["1990"],second_df["2000"],second_df["2005"],second_df["2013"]],
               fill_color='lavender',
               align='left'))
])
fig.show()

third_df =df_km2[df_km2["Segment K-means"]==2]
fig = go.Figure(data=[go.Table(
    header=dict(values=df_colums,
                fill_color='lightblue',
                align='left'),
    cells=dict(values=[third_df["Country"],third_df["1980"],third_df["1990"],third_df["2000"],third_df["2005"],third_df["2013"]],
               fill_color='lavender',
               align='left'))
])
fig.show()
# colors = ['#DF2020', '#81DF20', '#2095DF','#df20d5','#72806e']
# df['c'] = df_km2["Segment K-means"].map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]})
# import matplotlib.pyplot as plt
# plt.scatter(df_km2["2010"], df_km2["2012"], c=df_km2.c, alpha = 0.6, s=10)
# plt.show()

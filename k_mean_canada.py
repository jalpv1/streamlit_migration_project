import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
from PIL import Image


import streamlit as st

import plotly.graph_objects as go
data_container = st.container()
# Streamlit app
df_c = pd.read_csv("Canada.csv", sep=",")
df_a = pd.read_csv("americabirthplace.csv", sep=",")

cluster_list = []
with st.sidebar:
    st.write("")

selected_country = st.sidebar.selectbox("Select country", ["Canada", "America"])

if selected_country == "Canada":
    df = df_c
    k=4
    cluster_list= ["0", "1", "2", "3"]
else:
    df = df_a
    k=3
    cluster_list= ["0", "1", "2"]

years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']

#df = pd.read_csv("Canada.csv", sep=",")
print(df.head())
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

kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
kmeans.fit(df_std)
df_km = df_n2.copy()
df_km2 = df_km.copy()
df_km["Segment K-means"] = kmeans.labels_
#df_km["Country"] = df["OdName"]
gr = df_km[df_km['Segment K-means'] == 0]
df_km_analisys = df_km.groupby(["Segment K-means"]).count()
df_km_analisys["Segment K-means"] = cluster_list

print(df_km_analisys)

df_km2["Country"] = df["OdName"]
df_km2["Segment K-means"] = kmeans.labels_


#dynamic_filters.display_filters(location='sidebar')

#dynamic_filters.display_df()
#filtered_df = dynamic_filters.filter(melted_df)




if selected_country == "Canada":

    fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                                   cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                          ])
    cz = df_km2[df_km2["Country"] == "Czech Republic"][: 30]
    print(cz)
    first_df = df_km2[df_km2["Segment K-means"] == 0]
    second_df = df_km2[df_km2["Segment K-means"] == 1]
    third_df = df_km2[df_km2["Segment K-means"] == 2]
    fourth_df = df_km2[df_km2["Segment K-means"] == 3]

    df_colums = list(["Country_Name", "1980", "1990", "2000", "2005", "2013"])
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(values=[first_df["Country"], first_df["1980"], first_df["1990"], first_df["2000"], first_df["2005"],
                           first_df["2013"]],
                   align='left'))
    ])
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(
            values=[second_df["Country"], second_df["1980"], second_df["1990"], second_df["2000"], second_df["2005"],
                    second_df["2013"]],
            align='left'))
    ])
    fig3 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(values=[third_df["Country"], third_df["1980"], third_df["1990"], third_df["2000"], third_df["2005"],
                           third_df["2013"]],
                   align='left'))
    ])

    fig4 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(
            values=[fourth_df["Country"], fourth_df["1980"], fourth_df["1990"], fourth_df["2000"], fourth_df["2005"],
                    fourth_df["2013"]],
            align='left'))
    ])

    fig5 = go.Figure(data=[go.Table(
        header=dict(values=["Segment K-means","N"],
                    align='left'),
        cells=dict(
            values=[df_km_analisys["Segment K-means"], df_km_analisys["1980"]],
            align='left'))
    ])
    image = Image.open('canada.jpg')

    general, tab1, tab2, tab3, tab4 = st.tabs(["general","1", "2", "3", "4"])
    with general:
        st.plotly_chart(fig5, theme="streamlit")
        st.image(image, caption='Canada')

    with tab1:
        st.plotly_chart(fig1, theme="streamlit")
    with tab2:
        st.plotly_chart(fig2, theme="streamlit")

    with tab3:
        st.plotly_chart(fig3, theme="streamlit")

    with tab4:
        st.plotly_chart(fig4, theme="streamlit")
else:
    cz = df_km2[df_km2["Country"] == "Czech Republic"][: 30]
    print(cz)

    fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                                   cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                          ])
    first_df = df_km2[df_km2["Segment K-means"] == 0]
    second_df = df_km2[df_km2["Segment K-means"] == 1]
    third_df = df_km2[df_km2["Segment K-means"] == 2]
    fourth_df = df_km2[df_km2["Segment K-means"] == 3]

    df_colums = list(["Country_Name", "1980", "1990", "2000", "2005", "2013"])
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(values=[first_df["Country"], first_df["1980"], first_df["1990"], first_df["2000"], first_df["2005"],
                           first_df["2013"]],
                   align='left'))
    ])
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(
            values=[second_df["Country"], second_df["1980"], second_df["1990"], second_df["2000"], second_df["2005"],
                    second_df["2013"]],
            align='left'))
    ])
    fig3 = go.Figure(data=[go.Table(
        header=dict(values=df_colums,
                    align='left'),
        cells=dict(values=[third_df["Country"], third_df["1980"], third_df["1990"], third_df["2000"], third_df["2005"],
                           third_df["2013"]],
                   align='left'))
    ])
    fig5 = go.Figure(data=[go.Table(
        header=dict(values=["Segment K-means", "N"],
                    align='left'),
        cells=dict(
            values=[df_km_analisys["Segment K-means"], df_km_analisys["1980"]],
            align='left'))
    ])

    general, tab1, tab2, tab3 = st.tabs(["general","1", "2", "3"])
    image = Image.open('america.jpg')

    with general:
        st.plotly_chart(fig5, theme="streamlit")
        st.image(image, caption='America')


    with tab1:
        st.plotly_chart(fig1, theme="streamlit")
    with tab2:
        st.plotly_chart(fig2, theme="streamlit")

    with tab3:
        st.plotly_chart(fig3, theme="streamlit")

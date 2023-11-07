from turtle import st
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv('americabirthplace.csv')
# print(df.head(10))
# countries = df.AreaName.unique().tolist()
# fig = px.pie(df,values='2011',names='AreaName')
# st.header("Donut chart")
# st.plotly_chart(fig)
years = ['1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
for year in years:
    df[year] = pd.to_numeric(df[year], errors='coerce')
df=df.dropna(axis=0)
filtered_df = df[df['DevName'].str.contains('More developed regions')]

sums = filtered_df.sum(numeric_only = True)[years]
#fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
sums.reset_index(drop=True)
print(sums)
sum_df = pd.DataFrame({'year':sums.keys(), 'count':sums.values})
print(sum_df['count'])
fig = px.line(sum_df, x='year', y="count")
st.plotly_chart(fig, theme="streamlit")


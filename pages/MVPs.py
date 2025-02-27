import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from urllib.request import Request, urlopen
import time

def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.title('Basketball MVP Stats Explorer')

#load all the MVPs
@st.cache_data
def load_data():
    mvp_rows=[]
    for i in range (1950, 2025):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_per_game.html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        df = html[0]
        df['Awards'] = df['Awards'].fillna('', inplace=False)
        for j in range(0, len(df['Awards'])):
            flag = any(x=='MVP-1' for x in df.iloc[j]['Awards'].split(','))
            if flag:
                mvp_rows.append(df.iloc[j])
                break
    mvpdf = pd.DataFrame(mvp_rows, index=range(len(mvp_rows)))
    return mvpdf

mvp_csv ='mvp.csv'
if os.path.exists(mvp_csv):
    mvpdf = pd.read_csv(mvp_csv)
else:
    df1 = load_data()
    df1.to_csv(mvp_csv, index=False)

st.dataframe(mvpdf.drop(columns=['Rk']))

#players with multiple mvps
print(mvpdf['Player'].value_counts())

rows=st.columns(2)
#visualize how many games were played by all MVPs during their respective seasons
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mvpdf['G'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Games Played')
ax.set_ylabel('Number of MVPs')
ax.set_title('Number of games played by MVPs in their respective seasons')
rows[0].pyplot(fig)

#visualize the number of MVPs by points per game
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(mvpdf['PTS'], bins=20, color='skyblue', edgecolor='black')
ax2.set_ylabel('Number of MVPs')
ax2.set_xlabel('Points per game')
ax2.set_title('Points per game by MVPs in their respective seasons')
rows[1].pyplot(fig2)

#visualize the number of MVPs by position
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1 = sns.countplot(x='Pos', data=mvpdf, palette='viridis')
ax1.set_title('MVPs by Position')
ax1.set_ylabel('Number of MVPs')
ax1.set_xlabel('Position')
st.pyplot(fig1)








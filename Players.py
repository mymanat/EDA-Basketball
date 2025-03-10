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

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('Basketball Statistics Explorer')

#sidebar to select the year
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2025))))

#function toload all the players froma  specific year with their team stats
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(get_html(url), header = 0)
    time.sleep(2)
    df = html[0]

    # url1 = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + ".html"
    # html1 = pd.read_html(get_html(url1), header = 0)
    # time.sleep(2)
    # if year<1971:

    # elif year>=1971 or year<2016:
    
    # else:


    return df

#load the players for that specific year and drop the rank column
df_players = load_data(selected_year)
df_players = df_players.drop(columns=['Rk'])

#function to load the mvp for that year
@st.cache_data
def load_mvp(df):
    df['Awards'] = df['Awards'].fillna('', inplace=False)
    for j in range(0, len(df['Awards'])):
        flag = any(x=='MVP-1' for x in df.iloc[j]['Awards'].split(','))
        if flag: 
            return df.iloc[j]

#load the mvp for that year and create a dataframe of that mvp series
mvp = load_mvp(df_players)
df_mvp = pd.DataFrame([mvp], index=[0])

#functio to load the defensive player of the year for that year
@st.cache_data
def load_dpoy(df):
    df['Awards'] = df['Awards'].fillna('', inplace=False)
    for j in range(0, len(df['Awards'])):
        flag = any(x=='DPOY-1' for x in df.iloc[j]['Awards'].split(','))
        if flag: 
            return df.iloc[j]

#load the dpoy for that year and create a dataframe of that dpoy series   
dpoy = load_dpoy(df_players)
df_dpoy = pd.DataFrame([dpoy], index=[0])

#function to load the rookie of the year for that year
@st.cache_data
def load_roy(df):
    df['Awards'] = df['Awards'].fillna('', inplace=False)
    for j in range(0, len(df['Awards'])):
        flag = any(x=='ROY-1' for x in df.iloc[j]['Awards'].split(','))
        if flag: 
            return df.iloc[j]

#load the roy for that year and create a dataframe of that roy series       
roy = load_roy(df_players)
df_roy = pd.DataFrame([roy], index=[0])

#function to load the all stars for that year
@st.cache_data
def load_as(df):
    asList = []
    df['Awards'] = df['Awards'].fillna('', inplace=False)
    for j in range(0, len(df['Awards'])):
        if 'AS' in df.iloc[j]['Awards']: 
            asList.append(df.iloc[j])
    as_df = pd.DataFrame(asList, index=range(len(asList)))
    return as_df

#load the all stars for that year
as_df = load_as(df_players)

#all the players of that season
st.header(f'**Players of the {selected_year} season**')
st.dataframe(df_players)

#find the biggest correlation between stats of All Players displayed in a heatmap
if st.button('Correlation Heatmap'):
    st.header(f'Correlation Heatmap of the {selected_year} NBA players statistics')
    df_players_numeric = df_players.select_dtypes(include=['number'])
    corr = df_players_numeric.corr()
    fig, ax = plt.subplots()
    plot = sns.heatmap(corr, cmap='coolwarm')
    st.pyplot(fig)

#display the all stars of that season
st.header(f'**All Stars of the {selected_year} season**')
st.dataframe(as_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for all stars and All Players
as_stats = as_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards'], axis=1).mean(axis=0)
all_stats = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards'], axis=1).mean(axis=0)

#visualize the all stars stats vs All Players stats
fig2, ax2 = plt.subplots()
ax2.bar(as_stats.index, as_stats, color='skyblue', label='All Stars')
ax2.bar(all_stats.index, all_stats, color='orange', label='All Players')
ax2.set_xlabel('Statistics')
ax2.set_ylabel('Average per game')
ax2.xaxis.set_tick_params(rotation=90)
ax2.set_title(f'Average per game statistics of {selected_year} All Stars vs {selected_year} All Players')
ax2.legend()
st.pyplot(fig2)

#display the mvp with its statistics
st.header(f'**MVP(Most Valuable Player) of the {selected_year} season**')
st.dataframe(df_mvp)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for mvp
mvp_stats = df_mvp.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards'], axis=1).mean(axis=0)

#visualize the mvp stats vs all-stars stats
fig3, ax3 = plt.subplots()
ax3.bar(mvp_stats.index, mvp_stats, color='skyblue', label='MVP')
ax3.bar(as_stats.index, as_stats, color='orange', label='All-Stars')
ax3.set_ylabel('Average per game')
ax3.set_xlabel('Statistics')
ax3.xaxis.set_tick_params(rotation=90)
ax3.set_title(f'Average per game statistics of the {selected_year} MVP vs {selected_year} All Stars')
ax3.legend()
st.pyplot(fig3)

#display the dpoy with its statistics
st.header(f'**DPOY(Defensive Player Of the Year) of the {selected_year} season**')
st.dataframe(df_dpoy)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for dpoy
dpoy_stats = df_dpoy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards'], axis=1).mean(axis=0)

#visualize the dpoy stats vs All Players stats
fig4, ax4 = plt.subplots()
ax4.bar(dpoy_stats.index, dpoy_stats, color='skyblue', label='DPOY')
ax4.bar(all_stats.index, all_stats, color='orange', label='All Players')
ax4.set_ylabel('Average per game')
ax4.set_xlabel('Statistics')
ax4.xaxis.set_tick_params(rotation=90)
ax4.set_title(f'Average per game statistics of the {selected_year} DPOY vs {selected_year} All Players')
ax4.legend()
st.pyplot(fig4)

#display the roy with its statistics
st.header(f'**ROY(Rookie Of the Year) of the {selected_year} season**')
st.dataframe(df_roy)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for roy
roy_stats = df_roy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards'] , axis=1).mean(axis=0)

#visualize the roy stats vs All Players stats 
fig5, ax5 = plt.subplots()
ax5.bar(roy_stats.index, roy_stats, color='skyblue', label='ROY')
ax5.bar(all_stats.index, all_stats, color='orange', label='All Players')
ax5.set_ylabel('Average per game')
ax5.set_xlabel('Statistics')
ax5.xaxis.set_tick_params(rotation=90)
ax5.set_title(f'Average  per game statistics of the {selected_year} ROY vs {selected_year} All Players')
ax5.legend()
st.pyplot(fig5)



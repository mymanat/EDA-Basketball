import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import Request, urlopen
import time
from constants import FIRST_YEAR_NBA, FIRST_YEAR_MVP, CURRENT_YEAR, PREVIOUS_YEAR


def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('Basketball Stats Explorer')
team1 = st.header("First Team selected:")
selected_year = st.selectbox('Year1', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))

#load all the players
@st.cache_data
def load_teams(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + ".html"
    html = pd.read_html(get_html(url), header = 0)
    time.sleep(2)
    return html

#load the data and remove the divisions in the rows
if selected_year < 1971:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team1 = st.selectbox('Team1', df2['Team'])
    selected_team1_stats1 = df1.loc[df1['Team'] == selected_team1]
    selected_team1_stats2 = df2.loc[df2['Team'] == selected_team1]
elif selected_year>= 1971 and selected_year< 2016:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year)[2].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team1 = st.selectbox('Team1', df3['Team'])
    if selected_team1 in df1['Eastern Conference'].values:
        selected_team1_stats1 = df1.loc[df1['Eastern Conference'] == selected_team1]
    else:
        selected_team1_stats1 = df2.loc[df2['Western Conference'] == selected_team1]
    selected_team1_stats2 = df3.loc[df3['Team'] == selected_team1]
else:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year)[4].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team1 = st.selectbox('Team1', df3['Team'])
    if selected_team1 in df1['Eastern Conference'].values:
        selected_team1_stats1 = df1.loc[df1['Eastern Conference'] == selected_team1]
    else:
        selected_team1_stats1 = df2.loc[df2['Western Conference'] == selected_team1]
    selected_team1_stats2 = df3.loc[df3['Team'] == selected_team1]

team2 = st.header("Second Team selected:")
selected_year2 = st.selectbox('Year2', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))

if selected_year2 < 1971:
    df1 = load_teams(selected_year2)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year2)[1].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team2 = st.selectbox('Team2', df2['Team'])
    selected_team2_stats1 = df1.loc[df1['Team'] == selected_team2]
    selected_team2_stats2 = df2.loc[df2['Team'] == selected_team2]
elif selected_year2>= 1971 and selected_year2< 2016:
    df1 = load_teams(selected_year2)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year2)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year2)[2].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team2 = st.selectbox('Team2', df3['Team'])
    if selected_team2 in df1['Eastern Conference'].values:
        selected_team2_stats1 = df1.loc[df1['Eastern Conference'] == selected_team2]
    else:
        selected_team2_stats1 = df2.loc[df2['Western Conference'] == selected_team2]
    selected_team2_stats2 = df3.loc[df3['Team'] == selected_team2]
else:
    df1 = load_teams(selected_year2)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year2)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year2)[4].loc[lambda d: pd.to_numeric(d['Rk'], errors='coerce').notna()]
    selected_team2 = st.selectbox('Team2', df3['Team'])
    if selected_team2 in df1['Eastern Conference'].values:
        selected_team2_stats1 = df1.loc[df1['Eastern Conference'] == selected_team2]
    else:
        selected_team2_stats1 = df2.loc[df2['Western Conference'] == selected_team2]
    selected_team2_stats2 = df3.loc[df3['Team'] == selected_team2]

if st.button('Compare'):
    st.header(f'{selected_year} {selected_team1}')
    st.dataframe(selected_team1_stats1)
    st.dataframe(selected_team1_stats2)
    st.header(f'{selected_year2} {selected_team2}')
    st.dataframe(selected_team2_stats1)
    st.dataframe(selected_team2_stats2)

    #vizualise team1 w/l% vs team w/L%
    team1_wl = pd.to_numeric(selected_team1_stats1['W/L%']).mean()
    team2_wl = pd.to_numeric(selected_team2_stats1['W/L%']).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f'{selected_team1}', f'{selected_team2}'], [team1_wl, team2_wl], color=['skyblue', 'orange'])
    ax.set_ylabel('W/L%')
    ax.set_title(f'{selected_year} {selected_team1} vs {selected_year2} {selected_team2}')
    ax.legend()
    st.pyplot(fig)

    #visualize team1 PS/G, PA/G vs team2 PS/G, PA/G
    selected_team1_stats1[['PS/G', 'PA/G']] = selected_team1_stats1[['PS/G', 'PA/G']].apply(pd.to_numeric)
    selected_team2_stats1[['PS/G', 'PA/G']] = selected_team2_stats1[['PS/G', 'PA/G']].apply(pd.to_numeric)
    team1_psg_pag = selected_team1_stats1[['PS/G', 'PA/G']].mean()
    team2_psg_pag = selected_team2_stats1[['PS/G', 'PA/G']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = team1_psg_pag.index
    x = np.arange(len(labels))*2
    width = 0.8
    ax.bar(x-width/2, team1_psg_pag, color='skyblue', label=f'{selected_team1}')
    ax.bar(x+width/2, team2_psg_pag, color='orange', label=f'{selected_team2}')
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.set_title(f'PS/G(Points Scored per game) and PA/G(Points conceided per game) for {selected_year} {selected_team1} vs {selected_year2} {selected_team2}')
    ax.legend()
    st.pyplot(fig)

    #visualize the team1 stats fg%,3p%,2p%,ft% vs team2 stats fg%,3p%,2p%,ft%
    team1_stats = selected_team1_stats2.drop(['Rk','Team','G', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB','TRB', 'AST', 'BLK', 'STL', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
    team2_stats = selected_team2_stats2.drop(['Rk','Team','G', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB','TRB', 'AST', 'BLK', 'STL', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
    labels = team1_stats.index
    x = np.arange(len(labels))*2
    width = 0.8
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x-width/2, team1_stats, color='skyblue', label=f'{selected_team1}')
    ax1.bar(x+width/2, team2_stats, color='orange', label=f'{selected_team2}')
    ax1.set_ylabel('Mean')
    ax1.set_xlabel('Stat')
    ax1.set_xticklabels(labels)
    ax1.set_xticks(x)
    ax1.set_title(f'{selected_year} {selected_team1} vs {selected_year2} {selected_team2}')
    ax1.legend()
    st.pyplot(fig1)

    #visualize the team1 stats trb, ast, blk vs team2 stats trb, ast, blk
    team1_stats2 = selected_team1_stats2.drop(['Rk','Team','G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'STL', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
    team2_stats2 = selected_team2_stats2.drop(['Rk','Team','G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'STL', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
    labels = team1_stats2.index
    x = np.arange(len(labels))*2
    width = 0.8
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(x-width/2, team1_stats2, color='skyblue', label=f'{selected_team1}')
    ax2.bar(x+width/2, team2_stats2, color='orange', label=f'{selected_team2}')
    ax2.set_ylabel('Mean')
    ax2.set_xlabel('Stat')
    ax2.set_xticklabels(labels)
    ax2.set_xticks(x)
    ax2.set_title(f'{selected_year} {selected_team1} vs {selected_year2} {selected_team2}')
    ax2.legend()
    st.pyplot(fig2)
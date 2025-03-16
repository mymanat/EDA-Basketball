import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from urllib.request import Request, urlopen
import time

team_mapping = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",

    # Defunct or Relocated Teams
    "VAN": "Vancouver Grizzlies",
    "SEA": "Seattle SuperSonics",
    "NJN": "New Jersey Nets",
    "NYN": "New York Nets",
    "NOH": "New Orleans Hornets",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "NOJ": "New Orleans Jazz",
    "SDC": "San Diego Clippers",
    "BUF": "Buffalo Braves",
    "KCK": "Kansas City Kings",
    "CIN": "Cincinnati Royals",
    "ROC": "Rochester Royals",
    "PHW": "Philadelphia Warriors",
    "SFW": "San Francisco Warriors",
    "FTW": "Fort Wayne Pistons",
    "STL": "St. Louis Hawks",
    "MLH": "Milwaukee Hawks",
    "TCB": "Tri-Cities Blackhawks",
    "SYR": "Syracuse Nationals",
    "BAL": "Baltimore Bullets",
    "WSB": "Washington Bullets",
    "CHS": "Chicago Stags",
    "AND": "Anderson Packers",
    "SHE": "Sheboygan Red Skins",
    "WAT": "Waterloo Hawks",
    "PRO": "Providence Steamrollers",
    "MNL": "Minneapolis Lakers",
    "CHA": "Charlotte Bobcats",
    "SDR": "San Diego Rockets",
    "KCO": "Kansas City-Omaha Kings",
    "CAP": "Capital Bullets",
    "CHP": "Chicago Packers",
    "CHZ": "Chicago Zephyrs",
    "2TM": "2TM",
    "3TM": "3TM",
    "4TM": "4TM",
    "5TM": "5TM"
}

def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('Basketball Statistics Explorer')

#function toload all the players froma  specific year with their team stats
@st.cache_data
def player_stats():
    players = []
    for i in range(1956,2025):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_per_game.html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        players.append(html[0])
        html[0].insert(0,'Year',i, True)
    df = pd.concat(players, ignore_index=True).drop('Rk', axis=1)
    df= df[df['Player'].str.contains('League Average') == False]
    return df

df1 = player_stats()
st.title('Player stats per game')
st.dataframe(df1)

@st.cache_data
def player_advanced_stats():
    players = []
    for i in range(1956,2025):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_advanced.html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        html[0].insert(0,'Year',i, True)
        players.append(html[0])
    df = pd.concat(players, ignore_index=True).drop('Rk', axis=1)
    df= df[df['Player'].str.contains('League Average') == False]
    return df.drop(['Age', 'Pos', 'G', 'MP', 'GS', 'Awards'], axis=1)

df2 = player_advanced_stats()
st.title('Player advanced stats per game')
st.dataframe(df2)

df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
df_players_combined['Team'] = df_players_combined['Team'].map(team_mapping)
st.title('Player stats per game + advanced stats per game')
st.dataframe(df_players_combined)

@st.cache_data
def team_stats():
    teams = []
    for i in range(1956,2025):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + ".html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        if i<1971:
            html[1].insert(0,'Year',i, True)
            teams.append(html[1])
        elif i>=1971 and i<2016:
            html[2].insert(0,'Year',i, True)
            teams.append(html[2])
        else:
            html[4].insert(0,'Year',i, True)
            teams.append(html[4])
    df = pd.concat(teams, ignore_index=True).drop('Rk', axis=1)
    df= df[df['Team'].str.contains('League Average') == False]
    df['Team']= df['Team'].str.replace('*','', regex=False)
    return df

df3 = team_stats()
st.title('Team stats per game')
st.dataframe(df3)

@st.cache_data
def team_advanced_stats():
    teams = []
    for i in range(1956,2025):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + ".html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        for table in html:
            if 'Offense Four Factors' in table.columns:
                table = table.dropna(axis=1, how='all')
                table.insert(0,'Year',i, True)
                teams.append(table)
                break
    df = pd.concat(teams, ignore_index=True)
    df.rename(columns={'Unnamed: 0': 'Rk', 'Unnamed: 1': 'Team', 'Unnamed: 2': 'Age', 'Unnamed: 3': 'W', 
    'Unnamed: 4': 'L', 'Unnamed: 5': 'PW', 'Unnamed: 6': 'PL', 'Unnamed: 7': 'MOV', 'Unnamed: 8': 'SOS', 
    'Unnamed: 9': 'SRS', 'Unnamed: 10': 'ORtg', 'Unnamed: 11': 'DRtg', 'Unnamed: 12': 'NRtg', 
    'Unnamed: 13': 'Pace', 'Unnamed: 14': 'FTr', 'Unnamed: 15': '3PAr', 'Unnamed: 16': 'TS%', 
    'Offense Four Factors': 'eFG%_O', 'Offense Four Factors.1': 'TOV%_O', 'Offense Four Factors.2': 'ORB%_O', 
    'Offense Four Factors.3': 'FG/FGA_O', 'Defense Four Factors': 'eFG%_D', 'Defense Four Factors.1': 'TOV%_D',
    'Defense Four Factors.2': 'ORB%_D', 'Defense Four Factors.3': 'FG/FGA_D', 'Unnamed: 28': 'Arena', 
    'Unnamed: 29': 'Attend.', 'Unnamed: 30': 'Attend./G'}, inplace=True)
    df= df[df['Team'].str.contains('League Average') == False]
    df= df[df['Team'].str.contains('Team') == False]
    df['Team']= df['Team'].str.replace('*','', regex=False)
    df.reset_index(drop=True, inplace=True)
    df.drop(['Arena', 'Attend.', 'Attend./G'], axis=1, inplace=True)
    return df

df4 = team_advanced_stats()
st.title('Team advanced stats per game')
st.dataframe(df4)

df_teams_combined = pd.merge(df3, df4, on=['Year', 'Team'], how='inner')
st.title('Team stats per game + advanced stats per game')
st.dataframe(df_teams_combined)

# df_teams_combined_final = pd.merge(df_teams_combined, df5, on=['Year', 'Team'], how='inner')
# st.title('Player stats per game + advanced stats per game + Team standings')
# st.dataframe(df_teams_combined_final)

df_teams_players_combined = pd.merge(df_players_combined, df_teams_combined, on=['Year', 'Team'], how='inner')
st.title('Player all stats + Team all stats')
st.dataframe(df_teams_players_combined)

missing_players_teams = df_players_combined[~df_players_combined.set_index(['Year', 'Team']).index.isin(
    df_teams_players_combined.set_index(['Year', 'Team']).index
)]

# for i in range(len(missing_players_teams)):
#     for j in range(len(df_teams_players_combined)):
#         if missing_players_teams[['Year', 'Player']].iloc[i].equals(df_teams_players_combined[['Year', 'Player']].iloc[j]):
#             st.dataframe(df_teams_players_combined.iloc[j])
#             st.dataframe(missing_players_teams.iloc[i])

matches = missing_players_teams.merge(df_teams_players_combined, on=['Year', 'Player'], how='inner')

# Display matches
st.dataframe(matches)




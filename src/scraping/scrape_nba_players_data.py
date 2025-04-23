import os
import streamlit as st
import pandas as pd
from urllib.request import Request, urlopen
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import TEAM_MAPPING, FIRST_YEAR_MVP, PREVIOUS_YEAR

def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('MVP Classifier')

#function to load all the players from a specific year with their team stats
@st.cache_data
def player_stats(start_year, end_year):
    players = []
    for i in range(start_year,end_year+1):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_per_game.html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        players.append(html[0])
        html[0].insert(0,'Year',i, True)
    df = pd.concat(players, ignore_index=True).drop('Rk', axis=1)
    df= df[df['Player'].str.contains('League Average') == False]
    return df

@st.cache_data
def player_advanced_stats(start_year,end_year):
    players = []
    for i in range(start_year,end_year+1):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_advanced.html"
        html = pd.read_html(get_html(url), header = 0)
        time.sleep(2)
        html[0].insert(0,'Year',i, True)
        players.append(html[0])
    df = pd.concat(players, ignore_index=True).drop('Rk', axis=1)
    df= df[df['Player'].str.contains('League Average') == False]
    return df.drop(['Age', 'Pos', 'G', 'MP', 'GS', 'Awards'], axis=1)

@st.cache_data
def team_stats(start_year,end_year):
    teams = []
    for i in range(start_year,end_year+1):
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

@st.cache_data
def team_advanced_stats(start_year,end_year):
    teams = []
    for i in range(start_year,end_year+1):
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

all_seasons_players_csv ='data/raw/all_seasons_players.csv'
if os.path.exists(all_seasons_players_csv):
    all_seasons_players = pd.read_csv(all_seasons_players_csv)
else:
    df1 = player_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df2 = player_advanced_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
    df_players_combined['Team'] = df_players_combined['Team'].map(TEAM_MAPPING)
    df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
    df3 = team_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df4 = team_advanced_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)

    df_teams_combined = pd.merge(df3, df4, on=['Year', 'Team'], how='inner')
    df_teams_combined.rename(columns={col: f'Team-{col}' for col in df_teams_combined.columns.to_list()[2:]}, inplace=True)

    df_teams_players_combined = pd.merge(df_players_combined, df_teams_combined, on=['Year', 'Team'], how='inner')

    missing_players_teams = df_players_combined[~df_players_combined.set_index(['Year', 'Team']).index.isin(
        df_teams_players_combined.set_index(['Year', 'Team']).index
    )]

    new_df = pd.merge(df_teams_players_combined, missing_players_teams, how='outer')

    for index,row in missing_players_teams.iterrows():
        year = row['Year']
        player = row['Player']
        team = row['Team']

        player_rows = new_df[(new_df['Year']==year)&(new_df['Player']==player)&(~new_df['Team'].isin(['2TM', '3TM', '4TM', '5TM']))]

        if not player_rows.empty:
            games = pd.to_numeric(player_rows['Player-G'], errors='coerce')
            if games.sum() == 0:
                continue
            weights = games / games.sum()

            team_stat_cols = [col for col in new_df.columns if col.startswith('Team-')]

            numeric_stats = player_rows[team_stat_cols].apply(pd.to_numeric, errors='coerce')
            weighted_avg = (numeric_stats.T * weights).T.sum().round(3)

            for col in team_stat_cols:
                new_df.loc[(new_df['Year']==year) & (new_df['Player']==player) & (new_df['Team']==team), col] = weighted_avg[col]

    cols_to_multiply = [
        'Player-FG%', 'Player-FT%', 'Player-2P%', 'Player-3P%', 'Player-TS%', 'Player-eFG%',
        'Team-FG%', 'Team-3P%', 'Team-2P%', 'Team-FT%', 'Team-TS%', 'Team-eFG%_O', 
        'Team-eFG%_D'
    ]
    new_df[cols_to_multiply] = new_df[cols_to_multiply].apply(pd.to_numeric, errors='coerce')
    new_df[cols_to_multiply] = new_df[cols_to_multiply].multiply(100).round(3)

    new_df.to_csv(all_seasons_players_csv, index=False)
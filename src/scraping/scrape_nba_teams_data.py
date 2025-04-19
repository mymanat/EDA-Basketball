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

#create a all_seasons_teams cv file to store data and to avoid making a lot of requests to the website
all_seasons_teams_csv = 'data/raw/all_seasons_teams.csv'
if os.path.exists(all_seasons_teams_csv):
    all_seasons_teams_df = pd.read_csv(all_seasons_teams_csv)
else:
    df1 = team_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df2 = team_advanced_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df_teams_combined = pd.merge(df1, df2, on=['Year', 'Team'], how='inner')
    cols_to_multiply = ['FG%', '3P%', '2P%', 'FT%', 'TS%', 'eFG%_O', 'eFG%_D']
    df_teams_combined[cols_to_multiply]=df_teams_combined[cols_to_multiply].multiply(100)
    #if the csv file does not exist, create it
    df_teams_combined.to_csv(all_seasons_teams_csv, index=False)
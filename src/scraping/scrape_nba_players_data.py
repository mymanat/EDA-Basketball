import os
import streamlit as st
import pandas as pd
from urllib.request import Request, urlopen
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import TEAM_MAPPING, FIRST_YEAR_MVP, PREVIOUS_YEAR
from src.utils import scrape

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('MVP Classifier')

all_seasons_players_csv ='data/raw/all_seasons_players.csv'
if os.path.exists(all_seasons_players_csv):
    all_seasons_players = pd.read_csv(all_seasons_players_csv)
else:
    df1 = scrape.player_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df2 = scrape.player_advanced_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
    df_players_combined['Team'] = df_players_combined['Team'].map(TEAM_MAPPING)
    df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
    df3 = scrape.team_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)
    df4 = scrape.team_advanced_stats(FIRST_YEAR_MVP,PREVIOUS_YEAR)

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
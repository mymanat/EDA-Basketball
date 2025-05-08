import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import FIRST_YEAR_NBA, PREVIOUS_YEAR
from src.utils import scrape

#create a all_seasons_teams cv file to store data and to avoid making a lot of requests to the website
all_seasons_teams_csv = 'data/raw/all_seasons_teams.csv'
if os.path.exists(all_seasons_teams_csv):
    all_seasons_teams_df = pd.read_csv(all_seasons_teams_csv)
else:
    df1 = scrape.team_stats(FIRST_YEAR_NBA,PREVIOUS_YEAR)
    df2 = scrape.team_advanced_stats(FIRST_YEAR_NBA,PREVIOUS_YEAR)
    df_teams_combined = pd.merge(df1, df2, on=['Year', 'Team'], how='inner')
    cols_to_multiply = ['FG%', '3P%', '2P%', 'FT%', 'TS%', 'eFG%_O', 'eFG%_D']
    df_teams_combined[cols_to_multiply]=df_teams_combined[cols_to_multiply].astype(float)
    df_teams_combined[cols_to_multiply]=df_teams_combined[cols_to_multiply].multiply(100)
    #if the csv file does not exist, create it
    df_teams_combined.to_csv(all_seasons_teams_csv, index=False)
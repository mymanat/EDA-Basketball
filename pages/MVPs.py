import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import TEAM_MAPPING, CURRENT_YEAR
from src.utils import scrape

st.title('Basketball MVP Stats Explorer')

df = pd.read_csv('data/raw/all_seasons_players.csv')

mvpdf = df[df['Player-Awards'].fillna('').str.contains(r'\bMVP-1\b')].set_index('Year')

st.dataframe(mvpdf)

rows=st.columns(2)
#visualize how many games were played by all MVPs during their respective seasons
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mvpdf['Player-G'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Games Played')
ax.set_ylabel('Number of MVPs')
ax.set_title('Number of games played by MVPs in their respective seasons')
rows[0].pyplot(fig)

#visualize the number of MVPs by points per game
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(mvpdf['Player-PTS'], bins=20, color='skyblue', edgecolor='black')
ax2.set_ylabel('Number of MVPs')
ax2.set_xlabel('Points per game')
ax2.set_title('Points per game by MVPs in their respective seasons')
rows[1].pyplot(fig2)

#visualize the number of MVPs by position
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1 = sns.countplot(x='Player-Pos', data=mvpdf, palette='viridis')
ax1.set_title('MVPs by Position')
ax1.set_ylabel('Number of MVPs')
ax1.set_xlabel('Position')
st.pyplot(fig1)


#Creating DF for current season
current_season_players_csv ='data/raw/current_season_players.csv'
if os.path.exists(current_season_players_csv):
    current_season_players = pd.read_csv(current_season_players_csv)
else:
    df1 = scrape.player_stats(CURRENT_YEAR,CURRENT_YEAR)
    df2 = scrape.player_advanced_stats(CURRENT_YEAR,CURRENT_YEAR)
    df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
    df_players_combined['Team'] = df_players_combined['Team'].map(TEAM_MAPPING)
    df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
    df3 = scrape.team_stats(CURRENT_YEAR,CURRENT_YEAR)
    df4 = scrape.team_advanced_stats(CURRENT_YEAR,CURRENT_YEAR)

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
    new_df.to_csv(current_season_players_csv, index=False)

st.title(f"{CURRENT_YEAR} season players")
st.dataframe(current_season_players)

current = current_season_players.drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True)
for i in range(0, len(current)):
    if current.loc[i, 'Player-FGA'] == 0:
        current.loc[i, 'Player-FG%'] = 0
        current.loc[i, 'Player-eFG%'] = 0
        current.loc[i, 'Player-FTr'] = 0
        current.loc[i, 'Player-3PAr'] = 0
    if current.loc[i, 'Player-2PA'] == 0:
        current.loc[i, 'Player-2P%'] = 0
    if current.loc[i, 'Player-3PA'] == 0:
        current.loc[i, 'Player-3P%'] = 0
    if current.loc[i, 'Player-FTA'] == 0:
        current.loc[i, 'Player-FT%'] = 0
    if current.loc[i, 'Player-FGA'] == 0 or current.loc[i, 'Player-FTA'] == 0:
        current.loc[i, 'Player-TS%'] = 0
    if (current.loc[i, 'Player-FGA'] == 0) and (current.loc[i, 'Player-FTA'] == 0) and (current.loc[i, 'Player-TOV'] == 0):
        current.loc[i, 'Player-TOV%'] = 0
    if current.loc[i, 'Player-MP'] == 0:
        current.loc[i, 'Player-PER'] = 0
        current.loc[i, 'Player-WS/48'] = 0
        current.loc[i, 'Player-AST%'] = 0
        current.loc[i, 'Player-STL%'] = 0
        current.loc[i, 'Player-TRB%'] = 0
        current.loc[i, 'Player-ORB%'] = 0
        current.loc[i, 'Player-DRB%'] = 0
        current.loc[i, 'Player-BLK%'] = 0
        current.loc[i, 'Player-OBPM'] = 0
        current.loc[i, 'Player-DBPM'] = 0
        current.loc[i, 'Player-BPM'] = 0
        current.loc[i, 'Player-USG%'] = 0
        current.loc[i, 'Player-VORP'] = 0

linear_regression_model = joblib.load('src/models/linear_regression_model.pkl')

current_reordered = current[df[df['Year'] >= 2014].drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True).columns]
current_reordered = current_reordered.apply(pd.to_numeric, errors='coerce')

pred_current_year = linear_regression_model.predict(current_reordered)
prob_current_year = linear_regression_model.predict_proba(current_reordered)[:,1]
st.write("Number of MVP predictions according to logistic regression:", sum(pred_current_year))
current_reordered['MVP Prediction'] = pred_current_year
current_reordered['MVP Prediction Probability'] = prob_current_year
current_season_players_with_preds = current_season_players.copy()
current_season_players_with_preds['MVP Prediction'] = pred_current_year
current_season_players_with_preds['MVP Prediction Probability'] = prob_current_year
top_mvp_candidates = current_season_players_with_preds[current_season_players_with_preds['MVP Prediction']==1].sort_values(by='MVP Prediction Probability', ascending=False )
st.dataframe(top_mvp_candidates)

#Visualize the top 10 MVP features
importances = linear_regression_model.coef_[0]
feature_names = df[df['Year'] >= 2014].drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True).columns
sorted_indices = np.argsort(importances)[-10:]
plt.subplots(figsize=(10, 5))
plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
plt.title('Top 10 Feature Importances to MVP Prediction')
st.pyplot(plt.gcf())







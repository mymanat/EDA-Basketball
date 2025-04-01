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
st.title('MVP Classifier')

#function toload all the players from a specific year with their team stats
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

all_seasons_players_csv ='all_seasons_players.csv'
if os.path.exists(all_seasons_players_csv):
    all_seasons_players = pd.read_csv(all_seasons_players_csv)
else:
    df1 = player_stats(1956,2024)
    df2 = player_advanced_stats(1956,2024)
    df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
    df_players_combined['Team'] = df_players_combined['Team'].map(team_mapping)
    df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
    df3 = team_stats(1956,2024)
    df4 = team_advanced_stats(1956,2024)
    df4 = team_advanced_stats(1956,2024)

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

    new_df['Player-Awards'] = new_df['Player-Awards'].fillna('').astype(str)
    new_df['MVP'] = new_df['Player-Awards'].apply(lambda x: 1 if 'MVP-1' in x.split(',') else 0)
    new_df.to_csv(all_seasons_players_csv, index=False)

st.title("All players since 1956")
st.dataframe(all_seasons_players)

trying = all_seasons_players[all_seasons_players['Year'] >= 1980].drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True)
for i in range(0, len(trying)):
    if trying.loc[i, 'Player-FGA'] == 0:
        trying.loc[i, 'Player-FG%'] = 0
        trying.loc[i, 'Player-eFG%'] = 0
        trying.loc[i, 'Player-FTr'] = 0
        trying.loc[i, 'Player-3PAr'] = 0
    if trying.loc[i, 'Player-2PA'] == 0:
        trying.loc[i, 'Player-2P%'] = 0
    if trying.loc[i, 'Player-3PA'] == 0:
        trying.loc[i, 'Player-3P%'] = 0
    if trying.loc[i, 'Player-FTA'] == 0:
        trying.loc[i, 'Player-FT%'] = 0
    if trying.loc[i, 'Player-FGA'] == 0 or trying.loc[i, 'Player-FTA'] == 0:
        trying.loc[i, 'Player-TS%'] = 0
    if (trying.loc[i, 'Player-FGA'] == 0) and (trying.loc[i, 'Player-FTA'] == 0) and (trying.loc[i, 'Player-TOV'] == 0):
        trying.loc[i, 'Player-TOV%'] = 0
    if trying.loc[i, 'Player-MP'] == 0:
        trying.loc[i, 'Player-PER'] = 0
        trying.loc[i, 'Player-WS/48'] = 0
        trying.loc[i, 'Player-AST%'] = 0
        trying.loc[i, 'Player-STL%'] = 0
        trying.loc[i, 'Player-TRB%'] = 0
        trying.loc[i, 'Player-ORB%'] = 0
        trying.loc[i, 'Player-DRB%'] = 0
        trying.loc[i, 'Player-BLK%'] = 0
        trying.loc[i, 'Player-OBPM'] = 0
        trying.loc[i, 'Player-DBPM'] = 0
        trying.loc[i, 'Player-BPM'] = 0
        trying.loc[i, 'Player-USG%'] = 0
        trying.loc[i, 'Player-VORP'] = 0


st.dataframe(trying)

X = trying.drop(['MVP'], axis=1)
y = trying['MVP']

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101, stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Logistic Regression")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("Random Forest Classifier")
print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))

#SVC
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train,y_train)
svc_pred = model.predict(X_test)
print("SVC")
print(classification_report(y_test, svc_pred))
print(confusion_matrix(y_test, svc_pred))

#XGBOOST
from xgboost import XGBClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
print("XGB Classifier")
print(classification_report(y_test, xgb_predict))
print(confusion_matrix(y_test, xgb_predict))

#OVERSAMPLING
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=101)
X_smote, y_smote = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_smote,y_smote,test_size=0.2, random_state=101)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Logistic Regression after over-sampling")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("Random Forest Classifier after over-sampling")
print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))

model_svc = SVC(probability=True)
model_svc.fit(X_train,y_train)
svc_pred = model.predict(X_test)
print("SVC after over-sampling")
print(classification_report(y_test, svc_pred))
print(confusion_matrix(y_test, svc_pred))

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
print("XGB Classifier after over-sampling")
print(classification_report(y_test, xgb_predict))
print(confusion_matrix(y_test, xgb_predict))

#Creating DF for current season
df1 = player_stats(2025,2025)
df2 = player_advanced_stats(2025,2025)
df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
df_players_combined['Team'] = df_players_combined['Team'].map(team_mapping)
df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
df3 = team_stats(2025,2025)
df4 = team_advanced_stats(2025,2025)
df4 = team_advanced_stats(2025,2025)

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

current = new_df.drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True)
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

st.write("Current Season")
st.dataframe(current)

current_reordered = current[['Year', 'Player-G', 'Player-MP', 'Player-FG', 'Player-FGA', 'Player-FG%', 
                'Player-FT', 'Player-FTA', 'Player-FT%', 'Player-TRB', 'Player-AST', 'Player-PF', 
                'Player-PTS', 'Player-ORB', 'Player-DRB', 'Player-STL', 'Player-BLK', 'Player-TOV', 
                'Player-3P', 'Player-3PA', 'Player-3P%', 'Player-2P', 'Player-2PA', 'Player-2P%', 
                'Player-eFG%', 'Player-PER', 'Player-TS%', 'Player-FTr', 'Player-OWS', 'Player-DWS', 
                'Player-WS', 'Player-WS/48', 'Player-AST%', 'Player-TRB%', 'Player-ORB%', 'Player-DRB%', 
                'Player-STL%', 'Player-BLK%', 'Player-OBPM', 'Player-DBPM', 'Player-BPM', 'Player-VORP', 
                'Player-TOV%', 'Player-USG%', 'Player-3PAr', 'Team-G', 'Team-FG', 'Team-FGA', 'Team-FG%', 
                'Team-3P', 'Team-3PA', 'Team-3P%', 'Team-2P', 'Team-2PA', 'Team-2P%', 'Team-FT', 
                'Team-FTA', 'Team-FT%', 'Team-ORB', 'Team-DRB', 'Team-TRB', 'Team-AST', 'Team-STL', 
                'Team-BLK', 'Team-TOV', 'Team-PF', 'Team-PTS', 'Team-Rk', 'Team-Age', 'Team-W', 
                'Team-L', 'Team-PW', 'Team-PL', 'Team-MOV', 'Team-SOS', 'Team-SRS', 'Team-ORtg', 
                'Team-DRtg', 'Team-NRtg', 'Team-Pace', 'Team-FTr', 'Team-3PAr', 'Team-TS%', 'Team-eFG%_O', 
                'Team-TOV%_O', 'Team-ORB%_O', 'Team-FG/FGA_O', 'Team-eFG%_D', 'Team-TOV%_D', 
                'Team-ORB%_D', 'Team-FG/FGA_D']]
current_reordered = current_reordered.apply(pd.to_numeric, errors='coerce')
st.dataframe(current_reordered)
pred_2025 = model_svc.predict(current_reordered)
prob_2025 = model_svc.predict_proba(current_reordered)[:,1]
st.write("Number of MVP predictions:", sum(pred_2025))

current_reordered['MVP Prediction'] = pred_2025
current_reordered['MVP Prediction Probability'] = prob_2025
top_mvp_candidates = current_reordered.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
st.dataframe(top_mvp_candidates)

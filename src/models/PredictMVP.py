import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import Request, urlopen
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import TEAM_MAPPING, FIRST_YEAR_MVP, CURRENT_YEAR, PREVIOUS_YEAR

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

all_seasons_players = pd.read_csv('data/raw/all_seasons_players.csv')
all_seasons_players['Player-Awards'] = all_seasons_players['Player-Awards'].fillna('').astype(str)
all_seasons_players['MVP'] = all_seasons_players['Player-Awards'].apply(lambda x: 1 if 'MVP-1' in x.split(',') else 0)

st.title("All players since 1956")
st.dataframe(all_seasons_players)

mvp_training_data = all_seasons_players[all_seasons_players['Year'] >= 1980].drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True)
for i in range(0, len(mvp_training_data)):
    if mvp_training_data.loc[i, 'Player-FGA'] == 0:
        mvp_training_data.loc[i, 'Player-FG%'] = 0
        mvp_training_data.loc[i, 'Player-eFG%'] = 0
        mvp_training_data.loc[i, 'Player-FTr'] = 0
        mvp_training_data.loc[i, 'Player-3PAr'] = 0
    if mvp_training_data.loc[i, 'Player-2PA'] == 0:
        mvp_training_data.loc[i, 'Player-2P%'] = 0
    if mvp_training_data.loc[i, 'Player-3PA'] == 0:
        mvp_training_data.loc[i, 'Player-3P%'] = 0
    if mvp_training_data.loc[i, 'Player-FTA'] == 0:
        mvp_training_data.loc[i, 'Player-FT%'] = 0
    if mvp_training_data.loc[i, 'Player-FGA'] == 0 or mvp_training_data.loc[i, 'Player-FTA'] == 0:
        mvp_training_data.loc[i, 'Player-TS%'] = 0
    if (mvp_training_data.loc[i, 'Player-FGA'] == 0) and (mvp_training_data.loc[i, 'Player-FTA'] == 0) and (mvp_training_data.loc[i, 'Player-TOV'] == 0):
        mvp_training_data.loc[i, 'Player-TOV%'] = 0
    if mvp_training_data.loc[i, 'Player-MP'] == 0:
        mvp_training_data.loc[i, 'Player-PER'] = 0
        mvp_training_data.loc[i, 'Player-WS/48'] = 0
        mvp_training_data.loc[i, 'Player-AST%'] = 0
        mvp_training_data.loc[i, 'Player-STL%'] = 0
        mvp_training_data.loc[i, 'Player-TRB%'] = 0
        mvp_training_data.loc[i, 'Player-ORB%'] = 0
        mvp_training_data.loc[i, 'Player-DRB%'] = 0
        mvp_training_data.loc[i, 'Player-BLK%'] = 0
        mvp_training_data.loc[i, 'Player-OBPM'] = 0
        mvp_training_data.loc[i, 'Player-DBPM'] = 0
        mvp_training_data.loc[i, 'Player-BPM'] = 0
        mvp_training_data.loc[i, 'Player-USG%'] = 0
        mvp_training_data.loc[i, 'Player-VORP'] = 0

st.dataframe(mvp_training_data)

X = mvp_training_data.drop(['MVP'], axis=1)
y = mvp_training_data['MVP']

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("Random Forest Classifier after over-sampling")
print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))
print(roc_auc_score(y_test, rfc.predict_proba(X_test)[:,1]))

model_svc = SVC(probability=True)
model_svc.fit(X_train,y_train)
svc_pred = model.predict(X_test)
print("SVC after over-sampling")
print(classification_report(y_test, svc_pred))
print(confusion_matrix(y_test, svc_pred))
print(roc_auc_score(y_test, model_svc.predict_proba(X_test)[:,1]))

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
print("XGB Classifier after over-sampling")
print(classification_report(y_test, xgb_predict))
print(confusion_matrix(y_test, xgb_predict))
print(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]))

#Creating DF for current season
current_season_players_csv ='current_season_players.csv'
if os.path.exists(current_season_players_csv):
    current_season_players = pd.read_csv(current_season_players_csv)
else:
    df1 = player_stats(CURRENT_YEAR,CURRENT_YEAR)
    df2 = player_advanced_stats(CURRENT_YEAR,CURRENT_YEAR)
    df_players_combined = pd.merge(df1, df2, on=['Year', 'Player', 'Team'], how='inner')
    df_players_combined['Team'] = df_players_combined['Team'].map(TEAM_MAPPING)
    df_players_combined.rename(columns={col: f'Player-{col}' for col in df_players_combined.columns.to_list()[4:]}, inplace=True)
    df3 = team_stats(CURRENT_YEAR,CURRENT_YEAR)
    df4 = team_advanced_stats(CURRENT_YEAR,CURRENT_YEAR)

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

current_reordered = current[X.columns]
current_reordered = current_reordered.apply(pd.to_numeric, errors='coerce')
st.dataframe(current_reordered)

# pred_current_year = model.predict(current_reordered)
# prob_current_year = model.predict_proba(current_reordered)[:,1]
# st.write("Number of MVP predictions according to logistic regression:", sum(pred_current_year))
# current_reordered['MVP Prediction'] = pred_current_year
# current_reordered['MVP Prediction Probability'] = prob_current_year
# top_mvp_candidates = current_reordered.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
# st.dataframe(top_mvp_candidates)
# importances = model.coef_[0]
# feature_names = X.columns
# sorted_indices = np.argsort(importances)[-10:]
# plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
# plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
# st.pyplot(plt.gcf())


# pred_current_year = rfc.predict(current_reordered)
# prob_current_year = rfc.predict_proba(current_reordered)[:,1]
# st.write("Number of MVP predictions according to RandomForestClassifier:", sum(pred_current_year))
# current_reordered['MVP Prediction'] = pred_current_year
# current_reordered['MVP Prediction Probability'] = prob_current_year
# top_mvp_candidates = current_reordered.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
# st.dataframe(top_mvp_candidates)
# importances = rfc.feature_importances_
# feature_names = X.columns
# sorted_indices = np.argsort(importances)[-10:]
# plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
# plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
# st.pyplot(plt.gcf())


pred_current_year = model_svc.predict(current_reordered)
prob_current_year = model_svc.predict_proba(current_reordered)[:,1]
st.write("Number of MVP predictions according to SVC:", sum(pred_current_year))
current_season_players['MVP Prediction'] = pred_current_year
current_season_players['MVP Prediction Probability'] = prob_current_year
top_mvp_candidates = current_season_players.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
st.dataframe(top_mvp_candidates)
importances = model.coef_[0]
feature_names = X.columns
sorted_indices = np.argsort(importances)[-10:]
plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
st.pyplot(plt.gcf())


# pred_current_year = xgb_model.predict(current_reordered)
# prob_current_year = xgb_model.predict_proba(current_reordered)[:,1]
# st.write("Number of MVP predictions according to XGBoost:", sum(pred_current_year))
# current_reordered['MVP Prediction'] = pred_current_year
# current_reordered['MVP Prediction Probability'] = prob_current_year
# top_mvp_candidates = current_reordered.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
# st.dataframe(top_mvp_candidates)
# importances = xgb_model.feature_importances_
# feature_names = X.columns
# sorted_indices = np.argsort(importances)[-10:]
# plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
# plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
# st.pyplot(plt.gcf())

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import TEAM_MAPPING, CURRENT_YEAR
from src.utils import scrape

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('MVP Classifier')

all_seasons_players = pd.read_csv('data/raw/all_seasons_players.csv')
all_seasons_players['Player-Awards'] = all_seasons_players['Player-Awards'].fillna('').astype(str)
all_seasons_players['MVP'] = all_seasons_players['Player-Awards'].apply(lambda x: 1 if 'MVP-1' in x.split(',') else 0)

st.title("All players since 1956")
st.dataframe(all_seasons_players)

#MVP training data 
mvp_training_data = all_seasons_players[all_seasons_players['Year'] >= 2014].drop(['Player', 'Age', 'Team', 'Player-Pos', 'Player-Awards', 'Player-GS', 'Team-MP'], axis=1).reset_index(drop=True)
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

import joblib

model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'src/models/linear_regression_model.pkl')
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

# pred_current_year = model_svc.predict(current_reordered)
# prob_current_year = model_svc.predict_proba(current_reordered)[:,1]
# st.write("Number of MVP predictions according to SVC:", sum(pred_current_year))
# current_season_players['MVP Prediction'] = pred_current_year
# current_season_players['MVP Prediction Probability'] = prob_current_year
# top_mvp_candidates = current_season_players.sort_values(by='MVP Prediction Probability', ascending=False ).head(20)
# st.dataframe(top_mvp_candidates)
# importances = model.coef_[0]
# feature_names = X.columns
# sorted_indices = np.argsort(importances)[-10:]
# plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
# plt.yticks(range(len(sorted_indices)), feature_names[sorted_indices])
# st.pyplot(plt.gcf())

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
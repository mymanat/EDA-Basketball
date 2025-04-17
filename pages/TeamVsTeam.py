import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import FIRST_YEAR_NBA, PREVIOUS_YEAR

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('Basketball Stats Explorer')

teamsdf = pd.read_csv('data/raw/all_seasons_teams.csv')

team1 = st.header("First Team selected:")
selected_year1 = st.selectbox('Year1', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
selected_team1 = st.selectbox('Team1', teamsdf[teamsdf['Year']==selected_year1]['Team'])
selected_team1_stats = teamsdf.loc[teamsdf['Year']==selected_year1][teamsdf['Team']==selected_team1]

team2 = st.header("Second Team selected:")
selected_year2 = st.selectbox('Year2', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
selected_team2 = st.selectbox('Team2', teamsdf[teamsdf['Year']==selected_year2]['Team'])
selected_team2_stats = teamsdf.loc[teamsdf['Year']==selected_year2][teamsdf['Team']==selected_team2]

if st.button('Compare Teams'):
    st.header(f'{selected_year1} {selected_team1}')
    st.dataframe(selected_team1_stats)
    st.header(f'{selected_year2} {selected_team2}')
    st.dataframe(selected_team2_stats)

    #vizualise team1 w/l% vs team w/L%
    team1_wl = selected_team1_stats['W'].values[0]/selected_team1_stats['G'].values[0]
    team2_wl = selected_team2_stats['W'].values[0]/selected_team2_stats['G'].values[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f'{selected_team1}', f'{selected_team2}'], [team1_wl, team2_wl], color=['skyblue', 'orange'])
    ax.set_ylabel('W/L%')
    ax.set_title(f'{selected_year1} {selected_team1} vs {selected_year2} {selected_team2}')
    ax.legend()
    st.pyplot(fig)

    #visualize team1 PTS vs team2 PTS
    team1_pts = selected_team1_stats['PTS'].values[0]
    team2_pts = selected_team2_stats['PTS'].values[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([f'{selected_team1}', f'{selected_team2}'], [team1_pts, team2_pts], color=['skyblue', 'orange'])
    ax.set_ylabel('PTS')
    ax.set_title(f'{selected_year1} {selected_team1} vs {selected_year2} {selected_team2}')
    ax.legend()
    st.pyplot(fig)

    #visualize the team1 stats fg%,3p%,2p%,ft% vs team2 stats fg%,3p%,2p%,ft%
    team1_stats = selected_team1_stats[['FG%','3P%','2P%','FT%']]
    team2_stats = selected_team2_stats[['FG%','3P%','2P%','FT%']]
    labels = team1_stats.columns
    x = np.arange(len(labels))*2
    width = 0.8
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(x-width/2, team1_stats.values[0], color='skyblue', label=f'{selected_team1}')
    ax1.bar(x+width/2, team2_stats.values[0], color='orange', label=f'{selected_team2}')
    ax1.set_ylabel('Mean')
    ax1.set_xlabel('Stat')
    ax1.set_xticklabels(labels)
    ax1.set_xticks(x)
    ax1.set_title(f'{selected_year1} {selected_team1} vs {selected_year2} {selected_team2}')
    ax1.legend()
    st.pyplot(fig1)

    #visualize the team1 stats trb, ast, blk vs team2 stats trb, ast, blk
    team1_stats = selected_team1_stats[['TRB','AST','BLK']]
    team2_stats = selected_team2_stats[['TRB','AST','BLK']]
    labels = team1_stats.columns
    x = np.arange(len(labels))*2
    width = 0.8
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(x-width/2, team1_stats.values[0], color='skyblue', label=f'{selected_team1}')
    ax2.bar(x+width/2, team2_stats.values[0], color='orange', label=f'{selected_team2}')
    ax2.set_ylabel('Mean')
    ax2.set_xlabel('Stat')
    ax2.set_xticklabels(labels)
    ax2.set_xticks(x)
    ax2.set_title(f'{selected_year1} {selected_team1} vs {selected_year2} {selected_team2}')
    ax2.legend()
    st.pyplot(fig2)
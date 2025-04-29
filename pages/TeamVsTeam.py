import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import FIRST_YEAR_NBA, PREVIOUS_YEAR
from src.visualization import visual

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

    #visualize the team1 stats fg%,3p%,2p%,ft% vs team2 stats fg%,3p%,2p%,ft%
    fig1, ax1= visual.plot_stat_team_comparator((14,6), selected_team1_stats[['FG%','3P%','2P%','FT%']], selected_team2_stats[['FG%','3P%','2P%','FT%']], selected_team1, selected_team2, selected_year1, selected_year2, 'Stat', 'Mean')
    st.pyplot(fig1)

    #visualize the team1 stats trb, ast, blk vs team2 stats trb, ast, blk
    fig2, ax2= visual.plot_stat_team_comparator((14,6), selected_team1_stats[['PTS','TRB','AST','BLK']], selected_team2_stats[['PTS','TRB','AST','BLK']], selected_team1, selected_team2, selected_year1, selected_year2, 'Stat', 'Mean')
    st.pyplot(fig2)

    #visualize the team1 stats eFG%_O,TOV%_O, ORB%_O vs team2 stats eFG%_O,TOV%_O, ORB%_O
    fig3, ax3= visual.plot_stat_team_comparator((14,6), selected_team1_stats[['eFG%_O','TOV%_O','ORB%_O']], selected_team2_stats[['eFG%_O','TOV%_O','ORB%_O']], selected_team1, selected_team2, selected_year1, selected_year2, 'Stat', 'Mean')
    st.pyplot(fig3)

    #visualize the team1 stats eFG%_D,TOV%_D, ORB%_D vs team2 stats eFG%_D,TOV%_D, ORB%_D
    fig4, ax4= visual.plot_stat_team_comparator((14,6), selected_team1_stats[['eFG%_D','TOV%_D','ORB%_D']], selected_team2_stats[['eFG%_D','TOV%_D','ORB%_D']], selected_team1, selected_team2, selected_year1, selected_year2, 'Stat', 'Mean')
    st.pyplot(fig4)
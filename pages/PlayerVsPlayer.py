import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.constants import FIRST_YEAR_NBA, PREVIOUS_YEAR

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('Basketball Stats Explorer')

playersdf = pd.read_csv('data/raw/all_seasons_players.csv')

player1 = st.header("First Player selected:")
selected_year1 = st.selectbox('Year1', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
selected_player1 = st.selectbox('Player1', playersdf[playersdf['Year']==selected_year1]['Player'])
selected_player1_stats = playersdf.loc[playersdf['Year']==selected_year1][playersdf['Player']==selected_player1]

player2 = st.header("Second Player selected:")
selected_year2 = st.selectbox('Year2', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
selected_player2 = st.selectbox('Player2', playersdf[playersdf['Year']==selected_year2]['Player'])
selected_player2_stats = playersdf.loc[playersdf['Year']==selected_year2][playersdf['Player']==selected_player2]

# create a radar chart for each player and display their respective accolades
row = st.columns(2)
if st.button('Compare Players'):
    row[0].header(f'{selected_year1} {selected_player1}')
    row[0].dataframe(selected_player1_stats)
    df1 = pd.DataFrame(dict(
        r=selected_player1_stats[['Player-PTS','Player-AST','Player-TRB','Player-2P','Player-3P','Player-FT','Player-STL','Player-BLK']].values[0],
        theta= selected_player1_stats[['Player-PTS','Player-AST','Player-TRB','Player-2P','Player-3P','Player-FT','Player-STL','Player-BLK']].columns.tolist()
    ))
    fig1 = px.line_polar(df1, r='r', theta='theta', line_close=True)
    fig1.update_traces(fill='toself')
    fig1.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 5]),
        )
    )
    row[0].plotly_chart(fig1)
    row[0].write(selected_player1_stats['Player-Awards'])

    row[1].header(f'{selected_year2} {selected_player2}')
    row[1].dataframe(selected_player2_stats)
    df2 = pd.DataFrame(dict(
        r=selected_player2_stats[['Player-PTS','Player-AST','Player-TRB','Player-2P','Player-3P','Player-FT','Player-STL','Player-BLK']].values[0],
        theta= selected_player2_stats[['Player-PTS','Player-AST','Player-TRB','Player-2P','Player-3P','Player-FT','Player-STL','Player-BLK']].columns.tolist()
    ))
    fig2 = px.line_polar(df2, r='r', theta='theta', line_close=True)
    fig2.update_traces(fill='toself')
    fig2.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 5])
        )
    )
    row[1].plotly_chart(fig2)
    row[1].write(selected_player2_stats['Player-Awards'])



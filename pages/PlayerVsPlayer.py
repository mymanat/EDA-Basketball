import streamlit as st
import pandas as pd
import plotly.express as px
from src.constants import FIRST_YEAR_NBA, PREVIOUS_YEAR

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('🏀 NBA Statistics Explorer')

st.markdown("""
    <style>
    h1 {color: #FF4500;}
    .stButton>button {background-color: #FF4500; color: white;}
    </style>
""", unsafe_allow_html=True)

playersdf = pd.read_csv('data/raw/all_seasons_players.csv')

volume_stats = ['Player-PTS','Player-AST','Player-TRB','Player-2P','Player-3P','Player-FT','Player-STL','Player-BLK']
efficiency_stats = ['Player-FG%','Player-FT%','Player-2P%','Player-3P%','Player-TS%','Player-eFG%','Player-USG%','Player-TOV%']

#display stats for player1
player1 = st.header("First Player selected:")
selected_year1 = st.selectbox('Year1', ['All Time']+list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
if selected_year1 == 'All Time':
    selected_player1 = st.selectbox('Player1', playersdf['Player'].unique())
    selected_player1_stats = playersdf.loc[playersdf['Player']==selected_player1]
else:
    selected_player1 = st.selectbox('Player1', playersdf[playersdf['Year']==selected_year1]['Player'].unique())
    selected_player1_stats = playersdf.loc[playersdf['Year']==selected_year1][playersdf['Player']==selected_player1]

#display stats for player2
player2 = st.header("Second Player selected:")
selected_year2 = st.selectbox('Year2', ['All Time']+list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))
if selected_year2 == 'All Time':
    selected_player2 = st.selectbox('Player2', playersdf['Player'].unique())
    selected_player2_stats = playersdf.loc[playersdf['Player']==selected_player2]
else:
    selected_player2 = st.selectbox('Player2', playersdf[playersdf['Year']==selected_year2]['Player'].unique())
    selected_player2_stats = playersdf.loc[playersdf['Year']==selected_year2][playersdf['Player']==selected_player2]

# create a radar chart for each player and display their respective accolades
col1, col2 = st.columns(2)

if st.button('Compare Players'):
    with col1:
        st.header(f'{selected_year1} {selected_player1}')
        st.dataframe(selected_player1_stats)
    if selected_year1 == 'All Time':
        df1 = pd.DataFrame(dict(r=selected_player1_stats[volume_stats].mean().values,theta= volume_stats))
        df2 = pd.DataFrame(dict(r=selected_player1_stats[volume_stats].mean().values,theta= efficiency_stats))
    else:
        df1 = pd.DataFrame(dict(r=selected_player1_stats[volume_stats].values[0],theta= volume_stats))
        df2 = pd.DataFrame(dict(r=selected_player1_stats[efficiency_stats].values[0],theta= efficiency_stats))
    #display first radar chart
    fig1 = px.line_polar(df1, r='r', theta='theta', line_close=True)
    fig1.update_traces(fill='toself')
    fig1.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 5]),
        )
    )
    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    #display second radar chart
    fig2 = px.line_polar(df2, r='r', theta='theta', line_close=True)
    fig2.update_traces(fill='toself')
    fig2.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 100]),
        )
    )
    with col1:
        st.plotly_chart(fig2, use_container_width=True)
        st.write(selected_player1_stats['Player-Awards'])

    with col2:
        st.header(f'{selected_year2} {selected_player2}')
        st.dataframe(selected_player2_stats)
        
    if selected_year2 == 'All Time':
        df3 = pd.DataFrame(dict(r=selected_player2_stats[volume_stats].mean().values,theta= volume_stats))
        df4 = pd.DataFrame(dict(r=selected_player2_stats[efficiency_stats].mean().values,theta= efficiency_stats))
    else:
        df3 = pd.DataFrame(dict(r=selected_player2_stats[volume_stats].values[0],theta= volume_stats))
        df4 = pd.DataFrame(dict(r=selected_player2_stats[efficiency_stats].values[0],theta= efficiency_stats))
    
    fig3 = px.line_polar(df3, r='r', theta='theta', line_close=True)
    fig3.update_traces(fill='toself')
    fig3.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 5])
        )
    )
    with col2:
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line_polar(df4, r='r', theta='theta', line_close=True)
    fig4.update_traces(fill='toself')
    fig4.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 100]),
        )
    )
    with col2:
        st.plotly_chart(fig4, use_container_width=True)
        st.write(selected_player2_stats['Player-Awards'])
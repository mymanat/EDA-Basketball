import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import Request, urlopen
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.constants import PREVIOUS_YEAR, FIRST_YEAR_NBA
from src.visualization import visual

def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.set_page_config(page_title='Basketball Statistics Explorer', layout='wide')
st.title('Basketball Statistics Explorer')

#sidebar to select the year
selected_year = st.sidebar.selectbox('Year', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))

#function toload all the players froma  specific year with their team stats
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(get_html(url), header = 0)
    df = html[0]
    df = df.drop(df.loc[df['Player']=='League Average'].index)
    return df

#load the players for that specific year and drop the rank column
df_players = load_data(selected_year)
df_players = df_players.drop(columns=['Rk'])

#select the teams from the sidebar
selected_team = st.sidebar.multiselect('Team', df_players['Team'].unique(), df_players['Team'].unique())

#select the position from the sidebar
selected_pos = st.sidebar.multiselect('Position', df_players['Pos'].unique(), df_players['Pos'].unique())

#filtered data
df_players_filtered = df_players[(df_players['Team'].isin(selected_team))&(df_players['Pos'].isin(selected_pos))]

#function to load the mvp for that year
@st.cache_data
def load_mvp(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bMVP-1\b')]

#load the mvp for that year and create a dataframe of that mvp series
mvp_df = load_mvp(df_players)
mvp_df.reset_index(drop=True, inplace=True)

#function to load the defensive player of the year for that year
@st.cache_data
def load_dpoy(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bDPOY-1\b')]

#load the dpoy for that year and create a dataframe of that dpoy series   
dpoy_df = load_dpoy(df_players)
dpoy_df.reset_index(drop=True, inplace=True)

#function to load the rookie of the year for that year
@st.cache_data
def load_roy(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bROY-1\b')]

#load the roy for that year and create a dataframe of that roy series       
roy_df = load_roy(df_players)
roy_df.reset_index(drop=True, inplace=True)

#function to load the all stars for that year
@st.cache_data
def load_as(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bAS\b')]

#load the all stars for that year
as_df = load_as(df_players)
as_df.reset_index(drop=True, inplace=True)

#all the players of that season
st.header(f'**Players of the {selected_year} season**')
st.dataframe(df_players_filtered)

#find the biggest correlation between stats of All Stars displayed in a heatmap
if st.button('Correlation Heatmap'):
    st.header(f'Correlation Heatmap of the {selected_year} NBA players statistics')
    df_players_filtered_numeric = df_players_filtered.select_dtypes(include=['number'])
    corr = df_players_filtered_numeric.corr()
    fig, ax = plt.subplots()
    plot = sns.heatmap(corr, cmap='coolwarm', annot=True, annot_kws={"size": 3})
    st.pyplot(fig)

#display the all stars of that season
st.header(f'**All Stars of the {selected_year} season**')
st.dataframe(as_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for all stars and All Stars
if as_df.empty == False:
    if selected_year <= 1951:
        as_stats = as_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        as_stats_efficiency = as_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the all stars stats vs All Stars stats
        fig2, ax2 = visual.plot_player_type_comparator(as_stats, all_stats, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the all stars efficiency vs All Stars efficiency
        fig3, ax3 = visual.plot_player_type_comparator(as_stats_efficiency, all_stats_efficiency, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year < 1980 and selected_year > 1951:
        as_stats = as_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        as_stats_efficiency = as_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the all stars stats vs All Stars stats
        fig2, ax2 = visual.plot_player_type_comparator(as_stats, all_stats, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the all stars efficiency vs All Stars efficiency
        fig3, ax3 = visual.plot_player_type_comparator(as_stats_efficiency, all_stats_efficiency, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year >= 1980:
        as_stats = as_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        as_stats_efficiency = as_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the all stars stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(as_stats, all_stats, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the all stars efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(as_stats_efficiency, all_stats_efficiency, 'All Stars', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

#display the mvp with its statistics
st.header(f'**MVP(Most Valuable Player) of the {selected_year} season**')
st.dataframe(mvp_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for mvp
if mvp_df.empty == False:
    if selected_year <= 1951:
        mvp_stats = mvp_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        as_stats = as_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        mvp_stats_efficiency = mvp_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        as_stats_efficiency = as_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the MVP stats vs All Stars stats
        fig2, ax2 = visual.plot_player_type_comparator(mvp_stats, as_stats, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the MVP efficiency vs All Stars efficiency
        fig3, ax3 = visual.plot_player_type_comparator(mvp_stats_efficiency, as_stats_efficiency, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year < 1980 and selected_year > 1951:
        mvp_stats = mvp_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        as_stats = as_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        mvp_stats_efficiency = mvp_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = as_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the MVP stats vs All Stars stats
        fig2, ax2 = visual.plot_player_type_comparator(mvp_stats, as_stats, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the MVP efficiency vs All Stars efficiency
        fig3, ax3 = visual.plot_player_type_comparator(mvp_stats_efficiency, as_stats_efficiency, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year >= 1980:
        mvp_stats = mvp_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        as_stats = as_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        mvp_stats_efficiency = mvp_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
        as_stats_efficiency = as_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the MVP stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(mvp_stats, as_stats, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the MVP efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(mvp_stats_efficiency, as_stats_efficiency, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)


#display the dpoy with its statistics
st.header(f'**DPOY(Defensive Player Of the Year) of the {selected_year} season**')
st.dataframe(dpoy_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for dpoy
if dpoy_df.empty == False:
    if selected_year <= 1951:
        dpoy_stats = dpoy_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        dpoy_stats_efficiency = dpoy_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the DPY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(dpoy_stats, all_stats, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the DPOY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(as_stats_efficiency, all_stats_efficiency, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year < 1980 and selected_year > 1951:
        dpoy_stats = dpoy_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        dpoy_stats_efficiency = dpoy_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the DPOY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(dpoy_stats, all_stats, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the DPOY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(dpoy_stats_efficiency, all_stats_efficiency, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year >= 1980:
        dpoy_stats = dpoy_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        dpoy_stats_efficiency = dpoy_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the DPOY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(dpoy_stats, all_stats, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the DPOY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(dpoy_stats_efficiency, all_stats_efficiency, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

#display the roy with its statistics
st.header(f'**ROY(Rookie Of the Year) of the {selected_year} season**')
st.dataframe(roy_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for roy
if roy_df.empty == False:
    if selected_year <= 1951:
        roy_stats = roy_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        roy_stats_efficiency = roy_df.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the ROY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(roy_stats, all_stats, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the ROY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(roy_stats_efficiency, all_stats_efficiency, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year < 1980 and selected_year > 1951:
        roy_stats = roy_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG%', 'FT%'], axis=1).mean(axis=0)
        roy_stats_efficiency = roy_df.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the ROY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(roy_stats, all_stats, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the ROY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(roy_stats_efficiency, all_stats_efficiency, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)

    elif selected_year >= 1980:
        roy_stats = roy_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        all_stats = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
        roy_stats_efficiency = roy_df.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)
        all_stats_efficiency = df_players.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

        row = st.columns(2)

        #visualize the ROY stats vs All Players stats
        fig2, ax2 = visual.plot_player_type_comparator(roy_stats, all_stats, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[0].pyplot(fig2)

        #visualize the ROY efficiency vs All Players efficiency
        fig3, ax3 = visual.plot_player_type_comparator(roy_stats_efficiency, all_stats_efficiency, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
        row[1].pyplot(fig3)



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
    return df

#load the players for that specific year and drop the rank column
df_players = load_data(selected_year)
df_players = df_players.drop(columns=['Rk'])

#function to load the mvp for that year
@st.cache_data
def load_mvp(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bMVP-1\b')]

#load the mvp for that year and create a dataframe of that mvp series
df_mvp = load_mvp(df_players)
df_mvp.reset_index(drop=True, inplace=True)

#function to load the defensive player of the year for that year
@st.cache_data
def load_dpoy(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bDPOY-1\b')]

#load the dpoy for that year and create a dataframe of that dpoy series   
df_dpoy = load_dpoy(df_players)
df_dpoy.reset_index(drop=True, inplace=True)

#function to load the rookie of the year for that year
@st.cache_data
def load_roy(df):
    df['Awards'] = df['Awards'].fillna('')
    return df[df['Awards'].str.contains(r'\bROY-1\b')]

#load the roy for that year and create a dataframe of that roy series       
df_roy = load_roy(df_players)
df_roy.reset_index(drop=True, inplace=True)

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
st.dataframe(df_players)

#find the biggest correlation between stats of All Players displayed in a heatmap
if st.button('Correlation Heatmap'):
    st.header(f'Correlation Heatmap of the {selected_year} NBA players statistics')
    df_players_numeric = df_players.select_dtypes(include=['number'])
    corr = df_players_numeric.corr()
    fig, ax = plt.subplots()
    plot = sns.heatmap(corr, cmap='coolwarm', annot=True, annot_kws={"size": 3})
    st.pyplot(fig)

#display the all stars of that season
st.header(f'**All Stars of the {selected_year} season**')
st.dataframe(as_df)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for all stars and All Players
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
st.dataframe(df_mvp)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for mvp
mvp_stats = df_mvp.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards','FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
mvp_stats_efficiency = df_mvp.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

row = st.columns(2)

#visualize the mvp stats vs all-stars stats
fig4, ax4 = visual.plot_player_type_comparator(mvp_stats, as_stats, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
row[0].pyplot(fig4)

#visualize the mvp efficiency vs all-stars efficiency
fig5, ax5 = visual.plot_player_type_comparator(mvp_stats_efficiency, as_stats_efficiency, 'MVP', 'All Stars', selected_year, 'Statistics', 'Average per game')
row[1].pyplot(fig5)

#display the dpoy with its statistics
st.header(f'**DPOY(Defensive Player Of the Year) of the {selected_year} season**')
st.dataframe(df_dpoy)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for dpoy
dpoy_stats = df_dpoy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1).mean(axis=0)
dpoy_stats_efficiency = df_dpoy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

row = st.columns(2)

#visualize the dpoy stats vs All Players stats
fig6, ax6 = visual.plot_player_type_comparator(dpoy_stats, all_stats, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
row[0].pyplot(fig6)

#visualize the dpoy efficiency vs All Players efficiency
fig7, ax7 = visual.plot_player_type_comparator(dpoy_stats_efficiency, all_stats_efficiency, 'DPOY', 'All Players', selected_year, 'Statistics', 'Average per game')
row[1].pyplot(fig7)

#display the roy with its statistics
st.header(f'**ROY(Rookie Of the Year) of the {selected_year} season**')
st.dataframe(df_roy)

#drop the columns that are not needed for the mean calculation and 
# calculate the mean of each statistic for roy
roy_stats = df_roy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'] , axis=1).mean(axis=0)
roy_stats_efficiency = df_roy.drop(['Age','G','GS', 'MP','Player','Team','Pos', 'Awards', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'], axis=1).mean(axis=0)

row = st.columns(2)

#visualize the roy stats vs All Players stats 
fig8, ax8 = visual.plot_player_type_comparator(roy_stats, all_stats, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
row[0].pyplot(fig8)

#visualize the roy efficiency vs All Players efficiency
fig9, ax9 = visual.plot_player_type_comparator(roy_stats_efficiency, all_stats_efficiency, 'ROY', 'All Players', selected_year, 'Statistics', 'Average per game')
row[1].pyplot(fig9)



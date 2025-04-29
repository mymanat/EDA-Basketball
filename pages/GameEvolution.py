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

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('Basketball Stats Explorer')

#sidebar to select the year
selected_year = st.sidebar.selectbox('Year', list(reversed(range(FIRST_YEAR_NBA,PREVIOUS_YEAR+1))))

#load all the players
@st.cache_data
def load_teams(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + ".html"
    html = pd.read_html(get_html(url), header = 0)
    return html

#load the data and remove the divisions in the rows as appeared in the website
if selected_year < 1971:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1]
    merged_df = pd.merge(df1, df2, on='Team', how='inner')

    #create a 1 row of 2 columns to display the dataframes
    st.dataframe(merged_df)

    all_teams_stats = merged_df.drop(['G', 'W', 'SRS', 'L', 'GB'], axis=1)
    
    #highest correlation with winrate
    if st.button('Correlation Heatmap'):
        st.header(f'Correlation Heatmap of the {selected_year} NBA Teams statistics')
        all_teams_stats_numeric = all_teams_stats.apply(pd.to_numeric, errors='coerce').select_dtypes(include=['number']).dropna(axis=1, how='all')
        corr = all_teams_stats_numeric.corr()
        fig, ax = plt.subplots()
        plot = sns.heatmap(corr, cmap='coolwarm')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        plot = all_teams_stats_numeric.corr()['W/L%'].drop(index='W/L%', errors='ignore').sort_values().plot(kind='bar', ax=ax)
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Correlation with W/L%')
        ax.set_title(f'Correlation of different statistics impacting the W/L% for the {selected_year} season')
        st.pyplot(fig)

elif selected_year>=1971 and selected_year<2016:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year)[2]

    df1.rename(columns={'Eastern Conference':'Team'}, inplace=True)
    merged_df1 = pd.merge(df1, df3, on='Team', how='inner')
            
    df2.rename(columns={'Western Conference':'Team'}, inplace=True)
    merged_df2 = pd.merge(df2, df3, on='Team', how='inner')

    all_teams_stats = pd.concat([merged_df1,merged_df2], ignore_index=True, axis=0).drop(['G', 'W', 'SRS', 'L', 'GB'], axis=1)
    
    #create a 1 row of 2 columns to display the dataframes
    rows = st.columns(2)
    rows[0].dataframe(merged_df1)
    rows[1].dataframe(merged_df2)

    #highest correlation with winrate
    if st.button('Correlation Heatmap'):
        st.header(f'Correlation Heatmap of the {selected_year} NBA Teams statistics')
        all_teams_stats_numeric = all_teams_stats.apply(pd.to_numeric, errors='coerce').select_dtypes(include=['number']).dropna(axis=1, how='all')
        corr = all_teams_stats_numeric.corr()
        fig, ax = plt.subplots()
        plot = sns.heatmap(corr, cmap='coolwarm')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        plot = all_teams_stats_numeric.corr()['W/L%'].drop(index='W/L%', errors='ignore').sort_values().plot(kind='bar', ax=ax)
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Correlation with W/L%')
        ax.set_title(f'Correlation of different statistics impacting the W/L% for the {selected_year} season')
        st.pyplot(fig)

else:
    df1 = load_teams(selected_year)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df2 = load_teams(selected_year)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
    df3 = load_teams(selected_year)[4]

    df1.rename(columns={'Eastern Conference':'Team'}, inplace=True)
    merged_df1 = pd.merge(df1, df3, on='Team', how='inner')
    
    df2.rename(columns={'Western Conference':'Team'}, inplace=True)
    merged_df2 = pd.merge(df2, df3, on='Team', how='inner')
    
    all_teams_stats = pd.concat([merged_df1,merged_df2], ignore_index=True, axis=0).drop(['G', 'W', 'SRS', 'L', 'GB'], axis=1)
    
    #create a 1 row of 2 columns to display the dataframes
    rows = st.columns(2)
    rows[0].dataframe(merged_df1)
    rows[1].dataframe(merged_df2)

    #highest correlation with winrate
    if st.button('Correlation Heatmap'):
        st.header(f'Correlation Heatmap of the {selected_year} NBA Teams statistics')
        all_teams_stats_numeric = all_teams_stats.apply(pd.to_numeric, errors='coerce').select_dtypes(include=['number']).dropna(axis=1, how='all')
        corr = all_teams_stats_numeric.corr()
        fig, ax = plt.subplots()
        plot = sns.heatmap(corr, cmap='coolwarm')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        plot = all_teams_stats_numeric.corr()['W/L%'].drop(index='W/L%', errors='ignore').sort_values().plot(kind='bar', ax=ax)
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Correlation with W/L%')
        ax.set_title(f'Correlation of different statistics impacting the W/L% for the {selected_year} season')
        st.pyplot(fig)

df = pd.read_csv('data/raw/all_seasons_teams.csv')

grouped = df.drop('Team', axis=1).select_dtypes(include='number').groupby('Year').mean()

#plot the evolution of PPG
fig1, ax1 = visual.plot_stat_evolution((10,6), grouped, 'PTS', 'Average points scored per game by all teams since 1950','Year', 'Average points scored per game')
st.pyplot(fig1)

row1=st.columns(3)
row2=st.columns(3)
row3=st.columns(3)
row4=st.columns(3)

#row1 is all about 2P evolution
fig2, ax2 = visual.plot_stat_evolution((6,4), grouped, '2P', 'Average 2P made per game by all teams since 1950','Year', 'Average 2P made per game')
row1[0].pyplot(fig2)

fig3, ax3 = visual.plot_stat_evolution((6,4), grouped, '2PA', 'Average 2P attempted per game by all teams since 1950','Year', 'Average 2P attempted per game')
row1[1].pyplot(fig3)

fig4, ax4 = visual.plot_stat_evolution((6,4), grouped, '2P%', 'Average 2P% per game by all teams since 1950','Year', 'Average 2P% per game')
row1[2].pyplot(fig4)

#row2 is all about 3P evolution
fig5, ax5 = visual.plot_stat_evolution((6,4), grouped, '3P', 'Average 3P made per game by all teams since 1980','Year', 'Average 3P made per game')
row2[0].pyplot(fig5)

fig6, ax6 = visual.plot_stat_evolution((6,4), grouped, '3PA', 'Average 3P attempted per game by all teams since 1980','Year', 'Average 3P attempted per game')
row2[1].pyplot(fig6)

fig7, ax7 = visual.plot_stat_evolution((6,4), grouped, '3P%', 'Average 3P% per game by all teams since 1980','Year', 'Average 3P% per game')
row2[2].pyplot(fig7)

#row3 is all about FT(Free Throw) evolution
fig8, ax8 = visual.plot_stat_evolution((6,4), grouped, 'FT', 'Average FT made per game by all teams since 1950','Year', 'Average FT made per game')
row3[0].pyplot(fig8)

fig9, ax9 = visual.plot_stat_evolution((6,4), grouped, 'FTA', 'Average FT attempted per game by all teams since 1950','Year', 'Average FT attempted per game')
row3[1].pyplot(fig9)

fig10, ax10 = visual.plot_stat_evolution((6,4), grouped, 'FT%', 'Average FT% per game by all teams since 1950','Year', 'Average FT% per game')
row3[2].pyplot(fig10)

#TRB(Total Rebounds), AST(Assist), BLK(Block) per game evolution
fig11, ax11 = visual.plot_stat_evolution((6,4), grouped, 'TRB', 'Average TRB made per game by all teams since 1950','Year', 'Average TRB made per game')
row4[0].pyplot(fig11)

fig12, ax12 = visual.plot_stat_evolution((6,4), grouped, 'AST', 'Average AST per game by all teams since 1950','Year', 'Average AST per game')
row4[1].pyplot(fig12)

fig13, ax13 = visual.plot_stat_evolution((6,4), grouped, 'BLK', 'Average BLK per game by all teams since 1950','Year', 'Average BLK per game')
row4[2].pyplot(fig13)





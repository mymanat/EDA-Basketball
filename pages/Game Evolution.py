import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from urllib.request import Request, urlopen
import time


def get_html(url):
    req = Request(url, headers={ 'User-Agent': 'Mozilla/5.0'})
    res = urlopen(req)
    return res

st.set_page_config(page_title='Basketball Stats Explorer', layout='wide')
st.title('Basketball Stats Explorer')

#sidebar to select the year
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2025))))

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

#create a gameEvolution cv file to store data and to avoid making a lot of requests to the website
game_evolution_csv = 'gameEvolution.csv'
if os.path.exists(game_evolution_csv):
    game_evolution_df = pd.read_csv(game_evolution_csv)
else:
#show evolution of PS/G, PA/G in average by year for all teams across all seasons
    ppg_list = []
    opponent_ppg_list = []
    two_point_list = []
    two_point_attempt_list = []
    two_point_percentage_list = []
    three_point_list = []
    three_point_attempt_list = []
    three_point_percentage_list = []
    free_throw_list = []
    free_throw_attempt_list = []
    free_throw_percentage_list = []
    total_rebound_list = []
    assist_list = []
    steal_list = []
    block_list = []
    year_list = []
    #iterate through each year to collect the data, calculate its average
    # and append it to its respective lists
    #three different conditions because the data is structured differently for each bracket of years in the website
    for i in range(1950,2025):
        if i < 1971:
            df1 = load_teams(i)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
            df2 = load_teams(i)[1]
            df1['PS/G'] = pd.to_numeric(df1['PS/G'], errors='coerce')
            ppg_list.append(df1['PS/G'].mean())
            two_point_list.append(df2['2P'].mean())
            two_point_attempt_list.append(df2['2PA'].mean())   
            two_point_percentage_list.append(df2['2P%'].mean())
            three_point_list.append(df2['3P'].mean())
            three_point_attempt_list.append(df2['3PA'].mean())
            three_point_percentage_list.append(df2['3P%'].mean())
            free_throw_list.append(df2['FT'].mean())
            free_throw_attempt_list.append(df2['FTA'].mean())
            free_throw_percentage_list.append(df2['FT%'].mean())
            total_rebound_list.append(df2['TRB'].mean())
            assist_list.append(df2['AST'].mean())
            steal_list.append(df2['STL'].mean())
            block_list.append(df2['BLK'].mean())
            year_list.append(i)
        elif i>= 1971 and i< 2016:
            df1 = load_teams(i)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
            df2 = load_teams(i)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
            df3 = load_teams(i)[2]
            df1['PS/G'] = pd.to_numeric(df1['PS/G'], errors='coerce')
            df2['PS/G'] = pd.to_numeric(df2['PS/G'], errors='coerce')
            ppg_list.append((df1['PS/G'].mean()+ df2['PS/G'].mean())/2)
            two_point_list.append(df3['2P'].mean())
            two_point_attempt_list.append(df3['2PA'].mean())
            two_point_percentage_list.append(df3['2P%'].mean())
            three_point_list.append(df3['3P'].mean())
            three_point_attempt_list.append(df3['3PA'].mean())
            three_point_percentage_list.append(df3['3P%'].mean())
            free_throw_list.append(df3['FT'].mean())
            free_throw_attempt_list.append(df3['FTA'].mean())
            free_throw_percentage_list.append(df3['FT%'].mean())
            total_rebound_list.append(df3['TRB'].mean())
            assist_list.append(df3['AST'].mean())
            steal_list.append(df3['STL'].mean())
            block_list.append(df3['BLK'].mean())
            year_list.append(i)
        else:
            df1 = load_teams(i)[0].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
            df2 = load_teams(i)[1].loc[lambda d: pd.to_numeric(d['W'], errors='coerce').notna()]
            df3 = load_teams(i)[4]
            df1['PS/G'] = pd.to_numeric(df1['PS/G'], errors='coerce')
            df2['PS/G'] = pd.to_numeric(df2['PS/G'], errors='coerce')
            ppg_list.append((df1['PS/G'].mean()+ df2['PS/G'].mean())/2)
            two_point_list.append(df3['2P'].mean())
            two_point_attempt_list.append(df3['2PA'].mean())
            two_point_percentage_list.append(df3['2P%'].mean())
            three_point_list.append(df3['3P'].mean())
            three_point_attempt_list.append(df3['3PA'].mean())
            three_point_percentage_list.append(df3['3P%'].mean())
            free_throw_list.append(df3['FT'].mean())
            free_throw_attempt_list.append(df3['FTA'].mean())
            free_throw_percentage_list.append(df3['FT%'].mean())
            total_rebound_list.append(df3['TRB'].mean())
            assist_list.append(df3['AST'].mean())
            steal_list.append(df3['STL'].mean())
            block_list.append(df3['BLK'].mean())
            year_list.append(i)
    #if the csv file does not exist, create it
    game_evolution_df = pd.DataFrame({'Year': year_list, 'PPG': ppg_list, '2P': two_point_list, '2PA': two_point_attempt_list, '2P%': two_point_percentage_list, '3P': three_point_list, '3PA': three_point_attempt_list, '3P%': three_point_percentage_list, 'FT': free_throw_list, 'FTA': free_throw_attempt_list, 'FT%': free_throw_percentage_list, 'TRB': total_rebound_list, 'AST': assist_list, 'STL': steal_list, 'BLK': block_list})
    game_evolution_df.to_csv(game_evolution_csv, index=False)

#plot the evolution of PPG
fig, ax= plt.subplots(figsize=(10, 6))
ax.plot(game_evolution_df['Year'], game_evolution_df['PPG'], color='skyblue', linestyle='-')
ax.set_xlabel('Year')
ax.set_ylabel('Average points scored per game')
ax.set_title('Average points scored per game by all teams since 1950')
st.pyplot(fig)

row1=st.columns(3)
row2=st.columns(3)
row3=st.columns(3)
row4=st.columns(3)

#row1 is all about 2P evolution
fig2, ax2= plt.subplots(figsize=(6,4))
ax2.plot(game_evolution_df['Year'], game_evolution_df['2P'], color='skyblue', linestyle='-')
ax2.set_xlabel('Year')
ax2.set_ylabel('Average 2P made per game')
ax2.set_title('Average 2P made per game by all teams since 1950')
row1[0].pyplot(fig2)

fig3, ax3= plt.subplots(figsize=(6,4))
ax3.plot(game_evolution_df['Year'], game_evolution_df['2PA'], color='skyblue', linestyle='-')
ax3.set_xlabel('Year')
ax3.set_ylabel('Average 2P attempted per game')
ax3.set_title('Average 2P attempted per game by all teams since 1950')
row1[1].pyplot(fig3)

fig4, ax4= plt.subplots(figsize=(6,4))
ax4.plot(game_evolution_df['Year'], game_evolution_df['2P%'], color='skyblue', linestyle='-')
ax4.set_xlabel('Year')
ax4.set_ylabel('Average 2P% per game')
ax4.set_title('Average 2P% per game by all teams since 1950')
row1[2].pyplot(fig4)

#row2 is all about 3P evolution
fig5, ax5= plt.subplots(figsize=(6,4))
ax5.plot(game_evolution_df['Year'], game_evolution_df['3P'], color='skyblue', linestyle='-')
ax5.set_xlabel('Year')
ax5.set_ylabel('Average 3P made per game')
ax5.set_title('Average 3P made per game by all teams since 1980')
row2[0].pyplot(fig5)

fig6, ax6= plt.subplots(figsize=(6,4))
ax6.plot(game_evolution_df['Year'], game_evolution_df['3PA'], color='skyblue', linestyle='-')
ax6.set_xlabel('Year')
ax6.set_ylabel('Average 3P attempted per game')
ax6.set_title('Average 3P attempted per game by all teams since 1980')
row2[1].pyplot(fig6)

fig7, ax7= plt.subplots(figsize=(6,4))
ax7.plot(game_evolution_df['Year'], game_evolution_df['3P%'], color='skyblue', linestyle='-')
ax7.set_xlabel('Year')
ax7.set_ylabel('Average 3P% per game')
ax7.set_title('Average 3P% per game by all teams since 1980')
row2[2].pyplot(fig7)

#row3 is all about FT(Free Throw) evolution
fig8, ax8= plt.subplots()
ax8.plot(game_evolution_df['Year'], game_evolution_df['FT'], color='skyblue', linestyle='-')
ax8.set_xlabel('Year')
ax8.set_ylabel('Average FT made per game')
ax8.set_title('Average FT made per game by all teams since 1950')
row3[0].pyplot(fig8)

fig9, ax9= plt.subplots()
ax9.plot(game_evolution_df['Year'], game_evolution_df['FTA'], color='skyblue', linestyle='-')
ax9.set_xlabel('Year')
ax9.set_ylabel('Average FT attempted per game')
ax9.set_title('Average FT attempted per game by all teams since 1950')
row3[1].pyplot(fig9)

fig10, ax10= plt.subplots()
ax10.plot(game_evolution_df['Year'], game_evolution_df['FT%'], color='skyblue', linestyle='-')
ax10.set_xlabel('Year')
ax10.set_ylabel('Average FT% per game')
ax10.set_title('Average FT% per game by all teams since 1950')
row3[2].pyplot(fig10)

#TRB(Total Rebounds), AST(Assist), BLK(Block) per game evolution
fig11, ax11= plt.subplots()
ax11.plot(game_evolution_df['Year'], game_evolution_df['TRB'], color='skyblue', linestyle='-')
ax11.set_xlabel('Year')
ax11.set_ylabel('Average TRB made per game')
ax11.set_title('Average TRB made per game by all teams since 1950')
row4[0].pyplot(fig11)

fig12, ax12= plt.subplots()
ax12.plot(game_evolution_df['Year'], game_evolution_df['AST'], color='skyblue', linestyle='-')
ax12.set_xlabel('Year')
ax12.set_ylabel('Average AST per game')
ax12.set_title('Average AST per game by all teams since 1950')
row4[1].pyplot(fig12)

fig13, ax13= plt.subplots()
ax13.plot(game_evolution_df['Year'], game_evolution_df['BLK'], color='skyblue', linestyle='-')
ax13.set_xlabel('Year')
ax13.set_ylabel('Average BLK per game')
ax13.set_title('Average BLK per game by all teams since 1950')
row4[2].pyplot(fig13)







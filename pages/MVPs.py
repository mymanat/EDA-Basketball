import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Basketball MVP Stats Explorer')

df = pd.read_csv('data/raw/all_seasons_players.csv')

mvpdf = df[df['Player-Awards'].fillna('').str.contains(r'\bMVP-1\b')].set_index('Year')

st.dataframe(mvpdf)

rows=st.columns(2)
#visualize how many games were played by all MVPs during their respective seasons
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mvpdf['Player-G'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Games Played')
ax.set_ylabel('Number of MVPs')
ax.set_title('Number of games played by MVPs in their respective seasons')
rows[0].pyplot(fig)

#visualize the number of MVPs by points per game
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(mvpdf['Player-PTS'], bins=20, color='skyblue', edgecolor='black')
ax2.set_ylabel('Number of MVPs')
ax2.set_xlabel('Points per game')
ax2.set_title('Points per game by MVPs in their respective seasons')
rows[1].pyplot(fig2)

#visualize the number of MVPs by position
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1 = sns.countplot(x='Player-Pos', data=mvpdf, palette='viridis')
ax1.set_title('MVPs by Position')
ax1.set_ylabel('Number of MVPs')
ax1.set_xlabel('Position')
st.pyplot(fig1)        








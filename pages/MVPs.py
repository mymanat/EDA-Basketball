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

test = pd.read_csv('data/raw/all_seasons_players.csv')

newtest = test[test['Player']=='Stephen Curry']

st.dataframe(newtest)

st.write('Number of seasons in the NBA:', newtest['Year'].nunique())
st.write('Average PTS per game all time: ', newtest['Player-PTS'].mean())
st.write('Average AST per game all time: ', newtest['Player-AST'].mean())
st.write('Average TRB per game all time: ', newtest['Player-TRB'].mean())
st.write('Average 2P per game all time: ', newtest['Player-2P'].mean())
st.write('Average 3P per game all time: ', newtest['Player-3P'].mean())
st.write('Average FT per game all time: ', newtest['Player-FT'].mean())
st.write('Average STL per game all time: ', newtest['Player-STL'].mean())
st.write('Average BLK per game all time: ', newtest['Player-BLK'].mean())

st.write('Number of AS selections: ', newtest['Player-Awards'].str.count(r'\bAS\b').sum())
st.write('Number of MVP Awards: ', newtest['Player-Awards'].str.count(r'\bMVP-1\b').sum())
st.write('Number of DPOY Awards: ', newtest['Player-Awards'].str.count(r'\bDPOY-1\b').sum())
st.write('Number of ROY Awards: ', newtest['Player-Awards'].str.count(r'\bROY-1\b').sum())
st.write('Number of NBA First team selections : ', newtest['Player-Awards'].str.count(r'\bNBA1\b').sum())
st.write('Number of MVP Awards: ', newtest['Player-Awards'].str.count(r'\bMVP-1\b').sum())
st.write('Number of AS selections: ', newtest['Player-Awards'].str.count(r'\bAS\b').sum())







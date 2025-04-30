import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plot the evolution of a stat
def plot_stat_evolution(figsize, df, stat, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[stat], color='skyblue', linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig,ax

#bar plot to compare two teams stats
def plot_stat_team_comparator(figsize, team1_stats, team2_stats, team1_name, team2_name, year1, year2, xlabel, ylabel):
    labels = team1_stats.columns
    x = np.arange(len(labels))*2
    width = 0.8
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x-width/2, team1_stats.values[0], color='skyblue', label=f'{team1_name}')
    ax.bar(x+width/2, team2_stats.values[0], color='orange', label=f'{team2_name}')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.set_title(f'{year1} {team1_name} vs {year2} {team2_name}')
    ax.legend()
    return fig, ax

#heatmap of correlation between each stat for that specific year
def heatmap_corr(df, year):
    st.header(f'Correlation Heatmap of the {year} NBA Teams statistics')
    corr = df.corr()
    fig, ax = plt.subplots()
    plot = sns.heatmap(corr, cmap='coolwarm')
    st.pyplot(fig)

#plot the top stats that correlate with W/L%
def plot_winloss_corr(df, year):
    fig, ax = plt.subplots()
    plot = df.corr()['W/L%'].drop(index='W/L%', errors='ignore').sort_values().plot(kind='bar', ax=ax)
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Correlation with W/L%')
    ax.set_title(f'Correlation of different statistics impacting the W/L% for the {year} season')
    st.pyplot(fig)

#bar plot to compare stats from different types of players
def plot_player_type_comparator(group1_stats, group2_stats, group1_name, group2_name, year, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.bar(group1_stats.index, group1_stats, color='skyblue', label=f'{group1_name}')
    ax.bar(group2_stats.index, group2_stats, color='orange', label=f'{group2_name}')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_title(f'{year} {group1_name} vs {year} {group2_name}')
    ax.legend()
    return fig, ax
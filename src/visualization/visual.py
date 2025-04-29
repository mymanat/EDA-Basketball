import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#create a function to plot the evolution of a stat
def plot_stat_evolution(figsize, df, stat, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[stat], color='skyblue', linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig,ax

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
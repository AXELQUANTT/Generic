"""
Package devoted to group multiple plotting functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sns_lineplot(df:pd.DataFrame, x:str, y:str, gpby:str, title:str):
    """
    Given a dataframe df, it creates lineplots of x vs y, grouping
    the df by gpby column
    """
    
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(20,5))
    sns.lineplot(data=df, x=x, y=y, hue=gpby)
    plt.title(title)
    plt.show()
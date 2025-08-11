import os 
import pandas as pd
import numpy as np 

df = pd.read_excel('AI_Carpet_Roll_Photos.xlsx')

backing = df['Backing']
print(backing.head())




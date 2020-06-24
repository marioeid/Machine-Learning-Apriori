# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) # no tiltes in this dataset
transactions=[]

# aproiri expects a list of lists
print(len(dataset))
for i in range(0,len(dataset)):
     transactions.append([str(dataset.values[i,j]) for j in range(0,len(dataset.columns))])

#Training the apriori model 
from apyori import apriori
                           # (3*7)/7500
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2) 


# visualizing the result 
results=list(rules)
listRules = [list(results[i][0]) for i in range(0,len(results))]

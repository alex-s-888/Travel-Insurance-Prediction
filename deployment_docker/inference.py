"""
Executes batch prediction job.
"""

import sys

import numpy as np
import pandas as pd
from joblib import load

# Input data
infile = sys.argv[1]
# Placeholder for prediction
outfile = sys.argv[2]

print("Started prediction job. Input: " + infile)

df = pd.read_csv(infile)
df = df.drop(['Unnamed: 0'], axis=1)
df["ChronicDiseases"]= df["ChronicDiseases"].map({0: "No", 1: "Yes"})
df["TravelInsurance"]= df["TravelInsurance"].map({0: "not purchased", 1: "purchased"})

my_model = load("my_model.joblib")
prediction = my_model.predict(df[["AnnualIncome", "FamilyMembers", "Age"]])
with open(outfile, 'w') as my_output:
    for index, elem in np.ndenumerate(prediction):
        my_output.write(str(elem) + '\n')

print("Finished prediction job. Result: " + outfile)

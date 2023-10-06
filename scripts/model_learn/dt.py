import sys
import os
import yaml
import pickle

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file model \n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["train"]
p_seed = params["seed"]
p_max_depth = params["max_depth"]
p_spliter = params["splitter"]
p_min_samples_split = params["min_samples_split"]

df = pd.read_csv(f_input)

X = df.iloc[:,[0,1,3]]
y = df.iloc[:,2]

clf = DecisionTreeRegressor(max_depth=p_max_depth, random_state=p_seed, 
                            splitter=p_spliter, min_samples_split=p_min_samples_split)
clf.fit(X, y)

with open(f_output, "wb") as fd:
    pickle.dump(clf, fd)
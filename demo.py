import pandas as pd
import random
from propy import PyPro
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

random.seed(4)

positive_seqs = pd.read_csv("/zhouyuyang/ProteinFeatureEngineering/data/positive.csv")['sequence'].to_list()
negative_seqs = pd.read_excel("/zhouyuyang/ProteinFeatureEngineering/data/negative.xlsx")['Sequence'].to_list()

positive_seqs = [seq.replace('(','').replace(')','') for seq in positive_seqs if type(seq) is str]
negative_seqs = [seq.replace('(','').replace(')','') for seq in negative_seqs if type(seq) is str]

seqs = []
for seq in negative_seqs:
    seqs+=[seq[i:i+20] for i in range(0, len(seq), 20)]
negative_seqs = seqs

negative_seqs = [seq for seq in negative_seqs if len(seq)>1]

print("pos len:", len(positive_seqs))
print("neg len:", len(negative_seqs))

feature_list = None
X = []
y = []
for proteinsequence in positive_seqs:
    DesObject = PyPro.GetProDes(proteinsequence)  # construct a GetProDes object
    if feature_list is None:
        feature_list = DesObject.GetCTD().keys()
    X.append(list(DesObject.GetCTD().values()))  # calculate 147 CTD descriptors
    y.append(1)

for proteinsequence in negative_seqs:
    DesObject = PyPro.GetProDes(proteinsequence)  # construct a GetProDes object
    if feature_list is None:
        feature_list = DesObject.GetCTD().keys()
    X.append(list(DesObject.GetCTD().values()))  # calculate 147 CTD descriptors
    y.append(0)

regr = make_pipeline(StandardScaler(), SVR(kernel="linear",C=1.0, epsilon=0.2))
regr.fit(X, y)
print("score", regr.score(X, y))

params = zip(feature_list, regr.named_steps['svr'].coef_[0])
params=sorted(params, reverse=True, key=lambda x:(x[1], x[0]))
pd.DataFrame(params).to_csv("params.txt", index=None, header=None)

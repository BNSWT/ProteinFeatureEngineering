from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence

# download the protein sequence by uniprot id
proteinsequence = GetProteinSequence("P48039")

DesObject = PyPro.GetProDes(proteinsequence)  # construct a GetProDes object
print(DesObject.GetCTD())  # calculate 147 CTD descriptors
print(DesObject.GetAAComp())  # calculate 20 amino acid composition descriptors

# calculate 30 pseudo amino acid composition descriptors
paac = DesObject.GetPAAC(lamda=10, weight=0.05)

for i in paac:
    print(i, paac[i])
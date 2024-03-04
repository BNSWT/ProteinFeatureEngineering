"""
Apply L1-regularized SVC to perform embedded feature selection on descriptors with {+1/-1} response

Ref:	J. Bi, K. Bennett, M. Embrechts, C. Breneman, and M. Song J. Mach. Learn. Res. 3 1229 (2003)
	 	[Referred to in: Guyon & Elisseeff J. Mach. Learn. Res. 3 1157 (2003)]

IN:
<filename>         = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors
<seed>             = random seed for stratification and svc fitting
<alpha_MWU> 	   = significance threshold for Mann-Whitney U test above which descriptors will be eliminated
<rho_crit>         = critical rho_Spearman above which pairs of descriptors have one parter eliminated
<svc_tol>          = tolerance on svc stopping criterion
<svc_maxIter>      = maximum iterations in fitting svc
<n_CV>             = # rounds of stratified cross-validation to perform
<CVTestFraction>   = fraction of data to place in CV test set
<Cmin>             = minimum value of C to consider expressed as log10(C_min)
<Cmax>             = maximum value of C to consider expressed as log10(C_max)
<nC>               = # C values to consider on a log scale between C_min and C_max
<optFtol>          = tolerance in CV accuracy for optimization of L1-SVC C value
<nDummy>           = # dummy Gaussian distributed descriptors with which to dope descriptor matrix to test for probability of selecting features by chance 

OUT:

data_TRAIN_FILTERandZ__union_noflip.csv
Z_score_mean_std__union_noflip.csv
data_TRAIN_FILTERandZ__intersect_noflip.csv
Z_score_mean_std__intersect_noflip.csv
data_TRAIN_FILTERandZ__union_noflip_dummythresh.csv
Z_score_mean_std__union_noflip_dummythresh.csv


"""

## imports
import os, re, sys, time
import random, math

import numpy as np
import numpy.matlib

import scipy.optimize
import scipy.stats

import pickle as pkl

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.colors import Normalize

from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import grid_search


## classes
class MidpointNormalize(Normalize):		# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))




## methods

# usage
def _usage():
	print "USAGE: %s <filename> <svc_tol> <svc_maxIter> <n_CV> <seed> <Cmin> <Cmax> <nC>" % sys.argv[0]
	print "       <filename>         = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors"
	print "       <seed>             = random seed for stratification and svc fitting"
	print "       <alpha_MWU>        = significance threshold for Mann-Whitney U test above which descriptors will be eliminated"
	print "       <rho_crit>         = critical rho_Spearman above which pairs of descriptors have one parter eliminated"
	print "       <svc_tol>          = tolerance on svc stopping criterion"
	print "       <svc_maxIter>      = maximum iterations in fitting svc"
	print "       <n_CV>             = # rounds of stratified cross-validation to perform"
	print "       <CVTestFraction>   = fraction of data to place in CV test set"
	print "       <Cmin>             = minimum value of C to consider expressed as log10(C_min)"
	print "       <Cmax>             = maximum value of C to consider expressed as log10(C_max)"
	print "       <nC>               = # C values to consider on a log scale between C_min and C_max"
	print "       <optFtol>          = tolerance in CV accuracy for optimization of L1-SVC C value"
	print "       <nDummy>           = # dummy Gaussian distributed descriptors with which to dope descriptor matrix to test for probability of selecting features by chance "
	
	
def objFun(log10_C,svc,crossVal,data_desc,data_resp):
	
	C = math.pow(10,log10_C)
	
	svc.set_params(C=C)
	
	#scores = list()
	scores_CV = list()
	for idx_train, idx_validate in crossVal:
		svc = svc.fit(data_desc[idx_train], data_resp[idx_train])
		#scores.append(svc.score(data_desc[idx_train], data_resp[idx_train]))
		scores_CV.append(svc.score(data_desc[idx_validate], data_resp[idx_validate]))
	#scores_mean = np.mean(scores)
	#scores_std = np.std(scores)
	scores_CV_mean = np.mean(scores_CV)
	scores_CV_std = np.std(scores_CV)
	
	print("        C = %e, CV accuracy = %.3e +/- %.3e" % (C, scores_CV_mean, scores_CV_std))
	
	return -scores_CV_mean

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))



## main

# reading args and error checking
if len(sys.argv) != 14:
	_usage()
	sys.exit(-1)

infile = str(sys.argv[1])
seed = int(sys.argv[2])
alpha_MWU = float(sys.argv[3])
rho_crit = float(sys.argv[4])
svc_tol = float(sys.argv[5])
svc_maxIter = float(sys.argv[6])
n_CV = int(sys.argv[7])
CVTestFraction = float(sys.argv[8])
Cmin = float(sys.argv[9])
Cmax = float(sys.argv[10])
nC = int(sys.argv[11])
optFtol = float(sys.argv[12])
nDummy = int(sys.argv[13])




# setting seed
np.random.seed(seed)


# loading data
data_all=[]
with open(infile,'r') as fin:
	line = fin.readline()
	headers_all = line.strip().split(',')
	for line in fin:
		data_all.append(line.strip().split(','))

headers_file = headers_all[0]
data_file = [item[0] for item in data_all]

headers_index = headers_all[1]
data_index = [item[1] for item in data_all]

headers_resp = headers_all[2]
data_resp = [item[2] for item in data_all]
data_resp = [int(x) for x in data_resp]
data_resp = np.array(data_resp)

headers_desc = headers_all[3:]
data_desc = [item[3:] for item in data_all]
data_desc = [[float(y) for y in x] for x in data_desc]
data_desc = np.array(data_desc)

if np.std(data_resp) == 0:
	print("ERROR: response variables are all identical, there is no descrimination to be done here")
	sys.exit(-1)

if ((np.where((data_resp != 1) & (data_resp != -1)))[0]).size != 0:
	print("ERROR: Response variables are not exclusively +1 or -1; aborting.")
	sys.exit(-1)




# filtering
print("")
print("Commencing filtering of %d descriptors..." % data_desc.shape[1])
print("")







# (1) eliminating constant cols

print("  (1) Computing variance of each descriptor...")

idx_desc_KILL = np.where(np.std(data_desc,axis=0) == 0)[0]
idx_desc_KEEP = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KILL)]

headers_desc = [headers_desc[i] for i in idx_desc_KEEP]
data_desc = data_desc[:,idx_desc_KEEP]

print("      Eliminated %d zero variance descriptors, %d remain" % (len(idx_desc_KILL), data_desc.shape[1]))
print("")







# (2) Mann-Whitney U for distribution of each descriptor over +1 and -1 responses to identify those whose distribution differs significantly between +1 and -1 responses 
print("  (2) Computing Mann-Whitney U for distribution of each descriptor over +1 and -1 responses...")

MannWhitneyPval_array = np.empty(data_desc.shape[1])

for ii in range(0,data_desc.shape[1]):
	
	# - extracting descriptor 
	desc = data_desc[:,ii]
	
	# - extracting desc values for +1 and -1 responses
	desc_plus  = [desc[i] for i in range(0,len(desc)) if data_resp[i] == +1]
	desc_minus = [desc[i] for i in range(0,len(desc)) if data_resp[i] == -1]
	
	# - computing Mann-Whitney U p-value
	U, pval = scipy.stats.mannwhitneyu(desc_plus,desc_minus)
	
	# - storing
	MannWhitneyPval_array[ii] = pval
	
	# - printing
	#print("       Descriptor %d Mann-Whitney U pval = %12.5e" % (ii, pval))
	
'''
# plotting sorted Mann-Whitney U p-values
fig = plt.figure()
ax = plt.subplot(111)
ax.bar(range(0,len(Urd_array)), np.sort(MannWhitneyPval_array))
plt.xlabel('ordered binarized descriptor index')
plt.ylabel('Mann-Whitney p-value')
plt.savefig('MannWhitneyPval.png', bbox_inches='tight')
'''

# eliminating descriptors for which null hypothesis cannot be rejected at critical value of alpha_MWU
idx_desc_KILL = np.where(MannWhitneyPval_array > alpha_MWU)[0]
idx_desc_KEEP = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KILL)]

headers_desc = [headers_desc[i] for i in idx_desc_KEEP]
data_desc = data_desc[:,idx_desc_KEEP]
MannWhitneyPval_array = MannWhitneyPval_array[idx_desc_KEEP]

print("      Eliminated %d descriptors for which Mann-Whitney U null hypothesis cannot be rejected at critical value of %.2e, %d remain" % (len(idx_desc_KILL), alpha_MWU, data_desc.shape[1]))
print("")







# (3) Computing Spearman correlation coefficient between all pairs of descriptors to identify highly correlated descriptor pairs of which one member may be eliminated
print("  (3) Computing Spearman rho between all pairs of descriptors...")

# - computing all pairwise Spearman rhos
rho, pval = scipy.stats.spearmanr(data_desc, axis=0)

# - extracting those pairs with rho > rho_crit
rho_array = np.empty([0,], float)
pair_array = np.empty([0,2], int)

for ii in range(0,data_desc.shape[1]):
	for jj in range(ii+1,data_desc.shape[1]):
		if math.fabs(rho[ii,jj]) > rho_crit:
			rho_array = np.append(rho_array, rho[ii,jj])
			pair_array = np.vstack((pair_array, np.array([ii,jj])))

# - sorting pairs with rho > rho_crit into descending rho order
idx = np.argsort(rho_array, axis=0)
idx = idx[::-1]

rho_array = rho_array[idx]
pair_array = pair_array[idx,:]

# - while pairs remain in pair list, run through pairs in descending rho order to (i) randomly add one member of the pair to kill list, (ii) remove all pairs in pair list containing that descriptor
idx_desc_KILL = np.empty([0,], int)

while pair_array.shape[0] > 0:
	
	# -- determine descriptor in pair with higher Mann-Whitney U p-value and add to KILL list
	if MannWhitneyPval_array[pair_array[0,0]] > MannWhitneyPval_array[pair_array[0,1]]:
		tag = pair_array[0,0]
	else:
		tag = pair_array[0,1]

	idx_desc_KILL = np.append(idx_desc_KILL, tag)
	
	# -- eliminating all pairs containing the descriptor just added to KILL list
	mask = np.array(np.where(~np.any(pair_array == tag, axis=1)))[0,:]
	
	pair_array = pair_array[mask,:]
	rho_array = rho_array[mask]


# eliminating descriptors for which there is another with Spearman rho > rho_crit	
idx_desc_KILL = np.sort(idx_desc_KILL)
idx_desc_KEEP = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KILL)]

headers_desc = [headers_desc[i] for i in idx_desc_KEEP]
data_desc = data_desc[:,idx_desc_KEEP]

print("      Eliminated %d descriptors for which Spearman rho >= %.2e, %d remain" % (len(idx_desc_KILL), rho_crit, data_desc.shape[1]))
print("")







# (4) Z-scoring descriptors
print("  (4) Z-scoring descriptors...")

Z_means = np.mean(data_desc, axis=0)
Z_stds = np.std(data_desc, axis=0)

data_desc = data_desc - np.matlib.repmat(Z_means, data_desc.shape[0], 1)
data_desc = np.divide(data_desc, np.matlib.repmat(Z_stds, data_desc.shape[0], 1))

print("      Z-scored %d descriptors" % (data_desc.shape[1]))
print("")


# -- data file
outfile = infile[0:-4] + "_FILTERandZ__filterOnly.csv"

if len(headers_desc) > 0:

	with open(outfile,'w') as fout:
	
		fout.write(headers_file + ",")
		fout.write(headers_index + ",")
		fout.write(headers_resp + ",")
		for header in headers_desc:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for ii in range(0,data_desc.shape[0]):
			fout.write(data_file[ii] + ",")
			fout.write(data_index[ii] + ",")
			fout.write(str(data_resp[ii]) + ",")
			for jj in range(0,data_desc.shape[1]):
				fout.write(str(data_desc[ii][jj]) + ",")
			fout.seek(-1, os.SEEK_END)
			fout.truncate()
			fout.write("\n")

# -- Z-scoring info to apply to new data
with open("Z_score_mean_std__filterOnly.csv",'w') as fout:
	
	if len(headers_desc) > 0:
	
		for header in headers_desc:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for jj in range(0,data_desc.shape[1]):
			fout.write(str(Z_means[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

		for jj in range(0,data_desc.shape[1]):
			fout.write(str(Z_stds[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")	

# -- printing
print("      Wrote [%s, %s]" % (infile[0:-4] + "_FILTERandZ__filterOnly.csv","Z_score_mean_std__filterOnly.csv"))
print("")








# (5) Performing grid search for near-optimal C value for L1-SVC
print("  (5) Performing grid search for near-optimal C value for L1-SVC...")


# - defining linear SVC
#
#   \-> objective function to minimize: min(w,b) = P(f) + C*(1/n)*sum_i=1_n L(y_i,f(x_i))
#		where	f is the classifier
#				P(f) = penalty term = ||w||_k, where k=2 is usual choice, and k=1 enforces sparsity in feature space
#				L(y_i,f(x_i)) = loss term = [ max( 0, 1 - y_i*(w.x_i-b) ) ]^p = hinge loss (p=1), or squared-hinge loss (p=2)
#
#   \-> the parameter C controls the sparsity; the smaller C the greater weight on the penalty term relative to the loss term and fewer features selected
#
#	Refs:	http://scikit-learn.org/stable/modules/feature_selection.html
#			http://stackoverflow.com/questions/25042909/difference-between-penalty-and-loss-parameters-in-sklearn-linearsvc-library
#			http://scikit-learn.org/stable/modules/svm.html
#			http://en.wikipedia.org/wiki/Support_vector_machine
#			http://en.wikipedia.org/wiki/Hinge_loss
#
svc = svm.LinearSVC(C=100.0, loss='squared_hinge', penalty='l1', dual=False, tol=svc_tol, max_iter=svc_maxIter, random_state=seed)


# - defining CV subsets
crossVal = cross_validation.StratifiedShuffleSplit(data_resp, n_iter=n_CV, test_size=CVTestFraction, random_state=seed)


# - defining C range over which to search
Cempty = svm.l1_min_c(data_desc, data_resp, loss='squared_hinge')

if Cmax < Cempty:
	print("      ERROR: Maximum C value Cmax = %e is smaller than largest C value for which model is not empty Cempty = %e; aborting." % (Cmax, Cempty))
	sys.exit()
#if Cmin < Cempty:
#	print("      WARNING: Maximum C value Cmin = %f is smaller than largest C value for which model is not empty Cempty = %e" % (Cmin, Cempty))

C_array = np.logspace(math.log(Cmin,10), math.log(Cmax,10), nC)


# - performing automated grid search over C_array to seek C value that maximizes CV accuracy
fitMode = "manual"
if fitMode == "auto":	# automatic (cannot get number of non-zero (nnz) w elements nor training subset accuracy)

	params = {'C':C_array}
	clf = grid_search.GridSearchCV(estimator=svc, cv=crossVal, param_grid=params)
	clf.fit(data_desc, data_resp)        
	
	C_gsOpt = clf.best_estimator_.C
	
	#print clf.grid_scores_
	#print clf.best_score_                                  
	#print clf.best_estimator_
	#print clf.best_estimator_.C
	#print clf.best_estimator_.coef_

	coords_param =list()
	scores_CV_mean = list()
	scores_CV_std = list()
	for gridPoint in clf.grid_scores_:
		gridPointParams = gridPoint[0]
		gridPointParamsVals = list()
		for p in gridPointParams:
			gridPointParamsVals.append(gridPointParams[p])
		coords_param.append(gridPointParamsVals)
		scores_CV_mean.append(np.mean(gridPoint[2]))
		scores_CV_std.append(np.std(gridPoint[2]))

	coords_param = np.array(coords_param)
	scores_CV_mean = np.array(scores_CV_mean)
	scores_CV_std = np.array(scores_CV_std)

	plt.figure(1, figsize=(10, 8))
	plt.clf()
	plt.semilogx(C_array, scores_CV_mean, 'r', label='CV accuracy')
	plt.semilogx(C_array, scores_CV_mean + scores_CV_std, 'r--')
	plt.semilogx(C_array, scores_CV_mean - scores_CV_std, 'r--')
	plt.ylim(0, 1.05)
	plt.ylabel('CV accuracy')
	plt.xlabel('C')
	plt.title('CV accuracy')
	#plt.legend(loc='best')
	
	#plt.show()
	plt.savefig('CValGridSearch.png', bbox_inches='tight')
	
elif fitMode == "manual":
	
	scores_mean = list()
	scores_std = list()
	scores_CV_mean = list()
	scores_CV_std = list()
	nnz_mean = list()
	nnz_std = list()
	
	for C_val in C_array:
	
		# reparameterizing model
		svc.set_params(C=C_val)
	
		# training and cross-validation
		scores = list()
		scores_CV = list()
		nnz = list()
		for idx_train, idx_validate in crossVal:
			svc = svc.fit(data_desc[idx_train], data_resp[idx_train])
			scores.append(svc.score(data_desc[idx_train], data_resp[idx_train]))
			scores_CV.append(svc.score(data_desc[idx_validate], data_resp[idx_validate]))
			nnz.append( float((np.where(svc.coef_[0] != 0)[0]).shape[0]) )
		scores_mean.append(np.mean(scores))
		scores_std.append(np.std(scores))
		scores_CV_mean.append(np.mean(scores_CV))
		scores_CV_std.append(np.std(scores_CV))
		nnz_mean.append(np.mean(nnz))
		nnz_std.append(np.std(nnz))
		
		print("       C = %.2e, training subset accuracy = %.2e +/- %.2e, CV subset accuracy = %.2e +/- %.2e" % (C_val, scores_mean[-1], scores_std[-1], scores_CV_mean[-1], scores_CV_std[-1]))
	
	C_gsOpt = C_array[np.argmax(scores_CV_mean)]
	
	scores_mean = np.array(scores_mean)
	scores_std = np.array(scores_std)
	scores_CV_mean = np.array(scores_CV_mean)
	scores_CV_std = np.array(scores_CV_std)
	nnz_mean = np.array(nnz_mean)
	nnz_std = np.array(nnz_std)
	
	plt.figure(1, figsize=(10, 8))
	plt.clf()
	
	ax = plt.subplot(121)
	ax.set_xscale('log') 	#ax.set_xscale("log", nonposx='clip')
	plt.plot(C_array, scores_mean, 'b', label='training accuracy')
	plt.plot(C_array, scores_mean + scores_std, 'b--')
	plt.plot(C_array, scores_mean - scores_std, 'b--')
	plt.plot(C_array, scores_CV_mean, 'r', label='CV accuracy')
	plt.plot(C_array, scores_CV_mean + scores_CV_std, 'r--')
	plt.plot(C_array, scores_CV_mean - scores_CV_std, 'r--')
	plt.ylim(0, 1.05)
	plt.ylabel('CV accuracy')
	plt.xlabel('C')
	plt.title('CV accuracy')
	plt.legend(loc='best')
	
	ax = plt.subplot(122)
	plt.errorbar(nnz_mean, scores_mean, xerr=nnz_std, yerr=None, color='b', label='training accuracy')
	plt.plot(nnz_mean, scores_mean + scores_std, 'b--')
	plt.plot(nnz_mean, scores_mean - scores_std, 'b--')
	plt.errorbar(nnz_mean, scores_CV_mean, xerr=nnz_std, yerr=None, color='r', label='CV accuracy')
	plt.plot(nnz_mean, scores_CV_mean + scores_CV_std, 'r--')
	plt.plot(nnz_mean, scores_CV_mean - scores_CV_std, 'r--')
	plt.ylim(0, 1.05)
	plt.ylabel('CV accuracy')
	plt.xlabel('# descriptors')
	plt.title('CV accuracy')
	plt.legend(loc='best')
	
	#plt.show()
	plt.savefig('CValGridSearch.png', bbox_inches='tight')
	
	
	
		
	# -- printing
	with open("CValGridSearch.csv",'w') as fout:

		fout.write("C_array" + ",")
		fout.write("nnz_mean" + ",")
		fout.write("scores_mean" + ",")
		fout.write("scores_CV_mean" + ",")
		fout.write("scores_CV_std" + ",")

		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

		for ii in range(0,C_array.size):
			fout.write(str(C_array[ii]) + ",")
			fout.write(str(nnz_mean[ii]) + ",")
			fout.write(str(scores_mean[ii]) + ",")
			fout.write(str(scores_CV_mean[ii]) + ",")
			fout.write(str(scores_CV_std[ii]) + ",")

			fout.seek(-1, os.SEEK_END)
			fout.truncate()	
			fout.write("\n")

	print("     \-> Writing metrics to %s" % ("CValGridSearch.csv"))
	
	
else:

	print("ERROR: fitMode unrecognized; aborting")
	sys.exit()

print("      Best C value identified by grid search C_gsOpt = %e" % (C_gsOpt))
print("      Printing plot %s to file" % ("CValGridSearch.png"))
print("")







# (6) Optimizing L1-SVC C value initializing from best grid search result
print("  (6) Optimizing L1-SVC C value initializing from best grid search result C = %e..." % (C_gsOpt))

svc.set_params(C=C_gsOpt)

print("       Performing optimization of log10(C) to tolerance in CV accuracy of %.2e..." % (optFtol)) 

optResult = scipy.optimize.minimize(objFun, math.log(C_gsOpt,10), args=(svc,crossVal,data_desc,data_resp), method='Powell', options={'ftol': optFtol, 'disp': False})
if optResult.success == False:
	print optResult
	print("ERROR: Optimization of L1-SVC C value failed; aborting.")
	sys.exit()

print("       Optimization complete")

C_opt = math.pow(10,optResult.x)

print("      Numerical optimization identified C_opt = %e " % (C_opt))
print("")







# (7) Performing L1-SVC feature selection at optimal C
#	  Ref:	J. Bi, K. Bennett, M. Embrechts, C. Breneman, and M. Song J. Mach. Learn. Res. 3 1229 (2003)
print("  (7) Performing %d cross validation folds of L1-SVC feature selection at optimal C = %e..." % (n_CV, C_opt))
print("")

# - performing n_CV folds of cross validation with optimal C
svc.set_params(C=C_opt)

seed2 = np.random.randint(0, 4294967295)
crossVal = cross_validation.StratifiedShuffleSplit(data_resp, n_iter=n_CV, test_size=CVTestFraction, random_state=seed2)

scores = list()
scores_CV = list()
nnz = list()
activeFeatures = list()
activeFeatureWeights = list()
for idx_train, idx_validate in crossVal:
	svc = svc.fit(data_desc[idx_train], data_resp[idx_train])
	scores.append(svc.score(data_desc[idx_train], data_resp[idx_train]))
	scores_CV.append(svc.score(data_desc[idx_validate], data_resp[idx_validate]))
	nnz.append( float((np.where(svc.coef_[0] != 0)[0]).shape[0]) )
	activeFeatures.append( np.where(svc.coef_[0] != 0)[0] )
	activeFeatureWeights.append( svc.coef_[0,np.where(svc.coef_[0] != 0)[0]] )
scores_mean = np.mean(scores)
scores_std = np.std(scores)
scores_CV_mean = np.mean(scores_CV)
scores_CV_std = np.std(scores_CV)
nnz_mean = np.mean(nnz)
nnz_std = np.std(nnz)

print("      Averaging over folds:") 
print("       training accuracy = %.2e +/- %.2e" % (scores_mean, scores_std))
print("       CV accuracy       = %.2e +/- %.2e" % (scores_CV_mean, scores_CV_std))
print("       # active features = %.1f +/- %.1f" % (nnz_mean, nnz_std))
print("")

# - union, intersection, and sign flippers

# -- computing union of all features selected in n_CV folds of stratified cross-validation
activeFeatures_union = list()
for af in activeFeatures:
	activeFeatures_union = sorted(union( activeFeatures_union, af))	
	#print len(activeFeatures_union)
#print activeFeatures_union

# -- identifying number of folds in which each feature in the union appears
activeFeatures_union_counts = np.zeros(len(activeFeatures_union))
for ii in range(0,len(activeFeatures_union)):
	for jj in range(0,len(activeFeatures)):
		loc = np.where(activeFeatures[jj] == activeFeatures_union[ii])[0]
		if len(loc) > 0:
			activeFeatures_union_counts[ii] += 1
#print activeFeatures_union_counts

# -- identifying features whose weights flip sign
activeFeatures_union_weights = []
for idx in activeFeatures_union:
	idx_wts = []
	for ii in range(0,len(activeFeatures)):
		loc = np.where(activeFeatures[ii] == idx)[0]
		if len(loc) > 0:
			wt = float( activeFeatureWeights[ii][loc] )
			idx_wts.append(wt)
	activeFeatures_union_weights.append(idx_wts)
activeFeatures_union_flippers = []
for ii in range(0,len(activeFeatures_union_weights)):
	if activeFeatures_union_weights[ii][0] >= 0:
		sgn = +1
	else:
		sgn = -1
	for jj in range(1,len(activeFeatures_union_weights[ii])):
		if ( ( (sgn > 0) & (activeFeatures_union_weights[ii][jj] < 0) ) | ( (sgn < 0) & (activeFeatures_union_weights[ii][jj] >= 0) ) ):
			activeFeatures_union_flippers.append(activeFeatures_union[ii])
			break
#print activeFeatures_union_flippers

# -- computing intersection of all features selected in n_CV folds of stratified cross-validation
activeFeatures_intersect = list()
for ii in range(0,len(activeFeatures)):
	if ii == 0:
		activeFeatures_intersect = activeFeatures[ii]
	else:
		activeFeatures_intersect = sorted(intersect( activeFeatures_intersect, activeFeatures[ii] ))
	#print len(activeFeatures_intersect)
#print activeFeatures_intersect

# -- printing results
print("      Size of union of active features        = %d" % (len(activeFeatures_union))) 
print("      Size of intersection of active features = %d" % (len(activeFeatures_intersect)))
print("")
print("      \-> Number of weight sign flipping active features = %d" % (len(activeFeatures_union_flippers)))
print("")

# - eliminating sign flippers from union and intersection
activeFeatures_union_noflip = [x for x in activeFeatures_union if x not in activeFeatures_union_flippers]
activeFeatures_intersect_noflip = [x for x in activeFeatures_intersect if x not in activeFeatures_union_flippers]
print("      Size of union of non-flipping active features        = %d" % (len(activeFeatures_union_noflip))) 
print("      Size of intersection of non-flipping active features = %d" % (len(activeFeatures_intersect_noflip)))
print("")

# - doping descriptors with normally distributed random variables to empirically determine number of times expect to extract a variable whose correlation with the response is purely by chance

# -- generating Z-scores for nDummy dummy descriptors
dummy_desc = np.random.normal(loc=0.0, scale=1.0, size=(data_desc.shape[0],nDummy))
Z_means_dummy = np.mean(dummy_desc, axis=0)
Z_stds_dummy = np.std(dummy_desc, axis=0)
dummy_desc = dummy_desc - np.matlib.repmat(Z_means_dummy, dummy_desc.shape[0], 1)
dummy_desc = np.divide(dummy_desc, np.matlib.repmat(Z_stds_dummy, dummy_desc.shape[0], 1))

data_desc_DUMMYAUGMENT = np.concatenate((data_desc,dummy_desc), axis=1)

# -- repeating L1-SVC with n_CV folds of cross validation with optimal C with dummy augmented descriptor matrix
activeFeatures_DUMMYAUGMENT = list()
for idx_train, idx_validate in crossVal:
	svc = svc.fit(data_desc_DUMMYAUGMENT[idx_train], data_resp[idx_train])
	activeFeatures_DUMMYAUGMENT.append( np.where(svc.coef_[0] != 0)[0] )

# -- identifying number of folds in which each dummy variable appears
dummy_counts = np.zeros(nDummy)
for ii in range(0,nDummy):
	for jj in range(0,len(activeFeatures_DUMMYAUGMENT)):
		loc = np.where(activeFeatures_DUMMYAUGMENT[jj] == (data_desc.shape[1]+ii))[0]
		if len(loc) > 0:
			dummy_counts[ii] += 1

dummy_count_threshold = np.amax(dummy_counts)

print("      \-> Maximum number of times any of %d Gaussian random variable was selected within the %d CV folds = %d" % (nDummy, n_CV, dummy_count_threshold))
print("")

# -- identifying descriptors in activeFeatures_union that appear in more folds than dummy_count_threshold and do not weight flip
activeFeatures_union_noflip_dummythresh = list()
for ii in range(0,len(activeFeatures_union)):
	if ( ( len( np.where(activeFeatures_union_flippers == activeFeatures_union[ii])[0] ) == 0 ) & ( activeFeatures_union_counts[ii] > dummy_count_threshold ) ):
		activeFeatures_union_noflip_dummythresh.append(activeFeatures_union[ii])

print("      Size of union of dummy-thresholded non-flipping active features = %d" % (len(activeFeatures_union_noflip_dummythresh))) 
print("")







# (8) Writing non-flipping union and non-flipping intersection to file
print("  (8) Writing identified (i) non-flipping union, (ii) dummy-thresholded non-flipping union, and (iii) non-flipping intersection descriptors to file...")

# - non-flipping union

# -- extracting descriptors
idx_desc_KEEP = activeFeatures_union_noflip
idx_desc_KILL = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KEEP)]

headers_desc__union_noflip = [headers_desc[i] for i in idx_desc_KEEP]
data_desc__union_noflip = data_desc[:,idx_desc_KEEP]
Z_means__union_noflip = [Z_means[i] for i in idx_desc_KEEP]
Z_stds__union_noflip = [Z_stds[i] for i in idx_desc_KEEP]

# -- data file
outfile = infile[0:-4] + "_FILTERandZ__union_noflip.csv"

if len(headers_desc__union_noflip) > 0:

	with open(outfile,'w') as fout:
	
		fout.write(headers_file + ",")
		fout.write(headers_index + ",")
		fout.write(headers_resp + ",")
		for header in headers_desc__union_noflip:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for ii in range(0,data_desc__union_noflip.shape[0]):
			fout.write(data_file[ii] + ",")
			fout.write(data_index[ii] + ",")
			fout.write(str(data_resp[ii]) + ",")
			for jj in range(0,data_desc__union_noflip.shape[1]):
				fout.write(str(data_desc__union_noflip[ii][jj]) + ",")
			fout.seek(-1, os.SEEK_END)
			fout.truncate()
			fout.write("\n")

# -- Z-scoring info to apply to new data
with open("Z_score_mean_std__union_noflip.csv",'w') as fout:
	
	if len(headers_desc__union_noflip) > 0:
	
		for header in headers_desc__union_noflip:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for jj in range(0,data_desc__union_noflip.shape[1]):
			fout.write(str(Z_means__union_noflip[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

		for jj in range(0,data_desc__union_noflip.shape[1]):
			fout.write(str(Z_stds__union_noflip[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")	

# -- printing
print("      Wrote [%s, %s]" % (infile[0:-4] + "_FILTERandZ__union_noflip.csv","Z_score_mean_std__union_noflip.csv"))


# - dummy-thresholded non-flipping union

# -- extracting descriptors
idx_desc_KEEP = activeFeatures_union_noflip_dummythresh
idx_desc_KILL = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KEEP)]

headers_desc__union_noflip_dummythresh = [headers_desc[i] for i in idx_desc_KEEP]
data_desc__union_noflip_dummythresh = data_desc[:,idx_desc_KEEP]
Z_means__union_noflip_dummythresh = [Z_means[i] for i in idx_desc_KEEP]
Z_stds__union_noflip_dummythresh = [Z_stds[i] for i in idx_desc_KEEP]

# -- data file
outfile = infile[0:-4] + "_FILTERandZ__union_noflip_dummythresh.csv"

if len(headers_desc__union_noflip_dummythresh) > 0:

	with open(outfile,'w') as fout:
	
		fout.write(headers_file + ",")
		fout.write(headers_index + ",")
		fout.write(headers_resp + ",")
		for header in headers_desc__union_noflip_dummythresh:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for ii in range(0,data_desc__union_noflip_dummythresh.shape[0]):
			fout.write(data_file[ii] + ",")
			fout.write(data_index[ii] + ",")
			fout.write(str(data_resp[ii]) + ",")
			for jj in range(0,data_desc__union_noflip_dummythresh.shape[1]):
				fout.write(str(data_desc__union_noflip_dummythresh[ii][jj]) + ",")
			fout.seek(-1, os.SEEK_END)
			fout.truncate()
			fout.write("\n")

# -- Z-scoring info to apply to new data
with open("Z_score_mean_std__union_noflip_dummythresh.csv",'w') as fout:
	
	if len(headers_desc__union_noflip_dummythresh) > 0:
	
		for header in headers_desc__union_noflip_dummythresh:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for jj in range(0,data_desc__union_noflip_dummythresh.shape[1]):
			fout.write(str(Z_means__union_noflip_dummythresh[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

		for jj in range(0,data_desc__union_noflip_dummythresh.shape[1]):
			fout.write(str(Z_stds__union_noflip_dummythresh[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")	

# -- printing
print("      Wrote [%s, %s]" % (infile[0:-4] + "_FILTERandZ__union_noflip_dummythresh.csv","Z_score_mean_std__union_noflip_dummythresh.csv"))


# - non-flipping intersect

# -- extracting descriptors
idx_desc_KEEP = activeFeatures_intersect_noflip
idx_desc_KILL = [x for x in range(0,data_desc.shape[1]) if x not in set(idx_desc_KEEP)]

headers_desc__intersect_noflip = [headers_desc[i] for i in idx_desc_KEEP]
data_desc__intersect_noflip = data_desc[:,idx_desc_KEEP]
Z_means__intersect_noflip = [Z_means[i] for i in idx_desc_KEEP]
Z_stds__intersect_noflip = [Z_stds[i] for i in idx_desc_KEEP]

# -- data file
outfile = infile[0:-4] + "_FILTERandZ__intersect_noflip.csv"

if len(headers_desc__intersect_noflip) > 0:

	with open(outfile,'w') as fout:
	
		fout.write(headers_file + ",")
		fout.write(headers_index + ",")
		fout.write(headers_resp + ",")
		for header in headers_desc__intersect_noflip:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for ii in range(0,data_desc__intersect_noflip.shape[0]):
			fout.write(data_file[ii] + ",")
			fout.write(data_index[ii] + ",")
			fout.write(str(data_resp[ii]) + ",")
			for jj in range(0,data_desc__intersect_noflip.shape[1]):
				fout.write(str(data_desc__intersect_noflip[ii][jj]) + ",")
			fout.seek(-1, os.SEEK_END)
			fout.truncate()
			fout.write("\n")

# -- Z-scoring info to apply to new data
with open("Z_score_mean_std__intersect_noflip.csv",'w') as fout:
	
	if len(headers_desc__intersect_noflip) > 0:
	
		for header in headers_desc__intersect_noflip:
			fout.write(header + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")
	
		for jj in range(0,data_desc__intersect_noflip.shape[1]):
			fout.write(str(Z_means__intersect_noflip[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

		for jj in range(0,data_desc__intersect_noflip.shape[1]):
			fout.write(str(Z_stds__intersect_noflip[jj]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")	

# -- printing
print("      Wrote [%s, %s]" % (infile[0:-4] + "_FILTERandZ__intersect_noflip.csv","Z_score_mean_std__intersect_noflip.csv"))






print("")	
print("DONE!")
print("")


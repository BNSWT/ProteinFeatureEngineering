"""
Linear SVC to operate on {+1/-1} response data
Assumes all training and test descriptors have been Z-scored uniformly by same mean and stdev

IN:
<trainingData>     = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors
<testData>         = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors
<seed>             = random seed for stratification and svc fitting
<svc_tol>          = tolerance on svc stopping criterion
<svc_maxIter>      = maximum iterations in fitting svc
<n_CV>             = # rounds of stratified cross-validation to perform
<CVTestFraction>   = fraction of data to place in CV test set
<Cmin>             = minimum value of C to consider expressed as log10(Cmin)
<Cmax>             = maximum value of C to consider expressed as log10(Cmax)
<nC>               = # C values to consider on a log scale between Cmin and Cmax
<optFtol>          = tolerance in CV accuracy for optimization of L1-SVC C value

OUT:
training_weights.csv		= coefficients of descriptors in trained classifier
threshold_metrics.csv		= TP, FP, TN, FN and derived quantities as a function of the threshold distance from the margin
sorted_test_instances.csv	= test instances sorted by distance to margin and associated probability estimates for class membership

CValGridSearch.png			= grid search for optimal SVC parameter C
precision-recall.png		= precision (PPV) - recall (TPR) curve 
ROC.png						= ROC (TPR-FPR) curve

svc.pkl						= pickled trained classifier
svc.pkl_<idx>.npy			= pickled trained classifier ancillary files


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
from matplotlib.colors import Normalize

from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics


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
	print "USAGE: %s <trainingData> <testData> <seed> <svc_tol> <svc_maxIter> <n_CV> <CVTestFraction> <Cmin> <Cmax> <nC> <optFtol>" % sys.argv[0]
	print "       <trainingData>     = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors"
	print "       <testData>         = csv format: headerline; col 1 = descriptor_filename, col 2 = index, col 3 = response variable, col 4+ = descriptors"
	print "       <seed>             = random seed for stratification and svc fitting"
	print "       <svc_tol>          = tolerance on svc stopping criterion"
	print "       <svc_maxIter>      = maximum iterations in fitting svc"
	print "       <n_CV>             = # rounds of stratified cross-validation to perform"
	print "       <CVTestFraction>   = fraction of data to place in CV test set"
	print "       <Cmin>             = minimum value of C to consider expressed as log10(Cmin)"
	print "       <Cmax>             = maximum value of C to consider expressed as log10(Cmax)"
	print "       <nC>               = # C values to consider on a log scale between Cmin and Cmax"
	print "       <optFtol>          = tolerance in CV accuracy for optimization of L1-SVC C value"
	
	
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
	
	print("      C = %e, CV accuracy = %.3e +/- %.3e" % (C, scores_CV_mean, scores_CV_std))
	
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
if len(sys.argv) != 12:
	_usage()
	sys.exit(-1)

trainingData = str(sys.argv[1])
testData = str(sys.argv[2])
seed = int(sys.argv[3])
svc_tol = float(sys.argv[4])
svc_maxIter = float(sys.argv[5])
n_CV = int(sys.argv[6])
CVTestFraction = float(sys.argv[7])
Cmin = float(sys.argv[8])
Cmax = float(sys.argv[9])
nC = int(sys.argv[10])
optFtol = float(sys.argv[11])




# setting seed
np.random.seed(seed)


# loading data

# - loading training data
data_TRAIN=[]
with open(trainingData,'r') as fin:
	line = fin.readline()
	headers_TRAIN = line.strip().split(',')
	for line in fin:
		data_TRAIN.append(line.strip().split(','))

headers_file_TRAIN = headers_TRAIN[0]
data_file_TRAIN = [item[0] for item in data_TRAIN]

headers_index_TRAIN = headers_TRAIN[1]
data_index_TRAIN = [item[1] for item in data_TRAIN]

headers_resp_TRAIN = headers_TRAIN[2]
data_resp_TRAIN = [item[2] for item in data_TRAIN]
data_resp_TRAIN = [int(x) for x in data_resp_TRAIN]
data_resp_TRAIN = np.array(data_resp_TRAIN)

headers_desc_TRAIN = headers_TRAIN[3:]
data_desc_TRAIN = [item[3:] for item in data_TRAIN]
data_desc_TRAIN = [[float(y) for y in x] for x in data_desc_TRAIN]
data_desc_TRAIN = np.array(data_desc_TRAIN)

if np.std(data_resp_TRAIN) == 0:
	print("ERROR: Training data response variables are all identical, there is no descrimination to be done here")
	sys.exit(-1)

if ((np.where((data_resp_TRAIN != 1) & (data_resp_TRAIN != -1)))[0]).size != 0:
	print("ERROR: Training data response variables are not exclusively +1 or -1; aborting.")
	sys.exit(-1)


# - loading test data
data_TEST=[]
with open(testData,'r') as fin:
	line = fin.readline()
	headers_TEST = line.strip().split(',')
	for line in fin:
		data_TEST.append(line.strip().split(','))

headers_file_TEST = headers_TEST[0]
data_file_TEST = [item[0] for item in data_TEST]

headers_index_TEST = headers_TEST[1]
data_index_TEST = [item[1] for item in data_TEST]

headers_resp_TEST = headers_TEST[2]
data_resp_TEST = [item[2] for item in data_TEST]
data_resp_TEST = [int(x) for x in data_resp_TEST]
data_resp_TEST = np.array(data_resp_TEST)

headers_desc_TEST = headers_TEST[3:]
data_desc_TEST = [item[3:] for item in data_TEST]
data_desc_TEST = [[float(y) for y in x] for x in data_desc_TEST]
data_desc_TEST = np.array(data_desc_TEST)

if np.std(data_resp_TEST) == 0:
	print("ERROR: Test data response variables are all identical, there is no descrimination to be done here")
	sys.exit(-1)

if ((np.where((data_resp_TEST != 1) & (data_resp_TEST != -1)))[0]).size != 0:
	print("ERROR: Test data response variables are not exclusively +1 or -1; aborting.")
	sys.exit(-1)


# - consistency checking headers
if (headers_file_TRAIN == headers_file_TEST):
	headers_file = headers_file_TEST
else:
	print("ERROR: Col 1 header doesn't match between training and test data")
	sys.exit(-1)

if (headers_index_TRAIN == headers_index_TEST):
	headers_index = headers_index_TEST
else:
	print("ERROR: Col 2 header doesn't match between training and test data")
	sys.exit(-1)
	
if (headers_resp_TRAIN == headers_resp_TEST):
	headers_resp = headers_resp_TEST
else:
	print("ERROR: Col 3 header doesn't match between training and test data")
	sys.exit(-1)

if (len(headers_desc_TRAIN) != len(headers_desc_TEST)):
	print("ERROR: # descriptors differs between training and test data")
	sys.exit(-1)

for ii in range(0,len(headers_desc_TRAIN)):
	if (headers_desc_TRAIN[ii] != headers_desc_TEST[ii]):
		print("ERROR: name of descriptor %d differs between training and test data" % (ii+1))
		print("       training: %s" % (headers_desc_TRAIN[ii]))
		print("       test:     %s" % (headers_desc_TEST[ii]))
		sys.exit(-1)
headers_desc = headers_desc_TEST


# - error checking descriptors
print("")
print("  NOTE: Assuming training and test descriptors have been uniformly Z-scored...")
print("")
time.sleep(2)
'''
	Z_tol = 1E-10

	Z_means_TRAIN = np.mean(data_desc_TRAIN, axis=0)
	Z_stds_TRAIN = np.std(data_desc_TRAIN, axis=0)

	for ii in range(0,data_desc_TRAIN.shape[1]):
		if ( math.fabs(Z_means_TRAIN[ii]) > Z_tol ):
			print("ERROR: mean of descriptor %d in training data is not zero -- has this been Z-scored?" % (ii+1))
			sys.exit(-1)
		if ( math.fabs(Z_stds_TRAIN[ii]-1) > Z_tol ):
			print("ERROR: stdev of descriptor %d in training data is not unity -- has this been Z-scored?" % (ii+1))
			sys.exit(-1)

	Z_means_TEST = np.mean(data_desc_TEST, axis=0)
	Z_stds_TEST = np.std(data_desc_TEST, axis=0)

	for ii in range(0,data_desc_TEST.shape[1]):
		print math.fabs(Z_means_TEST[ii]) 
		if ( math.fabs(Z_means_TEST[ii]) > Z_tol ):
			print("ERROR: mean of descriptor %d in testing data is not zero -- has this been Z-scored?" % (ii+1))
			sys.exit(-1)
		if ( math.fabs(Z_stds_TEST[ii]-1) > Z_tol ):
			print("ERROR: stdev of descriptor %d in testing data is not unity -- has this been Z-scored?" % (ii+1))
			sys.exit(-1)
'''


# initializing svc 
#
#   \-> objective function to minimize: min(w,b) = P(f) + C*(1/n)*sum_i=1_n L(y_i,f(x_i))
#		where	f is the classifier
#				P(f) = penalty term = ||w||_k, where k=2 is usual choice, and k=1 enforces sparsity in feature space
#				L(y_i,f(x_i)) = loss term = [ max( 0, 1 - y_i*(w.x_i-b) ) ]^p = hinge loss is standard choice (p=1), or squared-hinge loss (p=2)
#
#   \-> the parameter C controls the sparsity; the smaller C the greater weight on the penalty term relative to the loss term and fewer features selected
#
#	Refs:	http://scikit-learn.org/stable/modules/feature_selection.html
#			http://stackoverflow.com/questions/25042909/difference-between-penalty-and-loss-parameters-in-sklearn-linearsvc-library
#			http://scikit-learn.org/stable/modules/svm.html
#			http://en.wikipedia.org/wiki/Support_vector_machine
#			http://en.wikipedia.org/wiki/Hinge_loss
#			http://scikit-learn.org/stable/modules/svm.html#svm-kernels
#
svc = svm.SVC(verbose=False, kernel='linear', C=1.0, tol=svc_tol, max_iter=svc_maxIter, random_state=seed)


# TRAINING CLASSIFIER
print("  == TRAINING CLASSIFIER on training data ==")
print("")

# - initializing CV subsets
crossVal = cross_validation.StratifiedShuffleSplit(data_resp_TRAIN, n_iter=n_CV, test_size=CVTestFraction, random_state=seed)


# - performing grid search for C range over defined range by stratified cross validation over training data
print("  (1) Performing grid search for near-optimal C value...")

C_array = np.logspace(math.log(Cmin,10), math.log(Cmax,10), nC)

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
		svc = svc.fit(data_desc_TRAIN[idx_train], data_resp_TRAIN[idx_train])
		scores.append(svc.score(data_desc_TRAIN[idx_train], data_resp_TRAIN[idx_train]))
		scores_CV.append(svc.score(data_desc_TRAIN[idx_validate], data_resp_TRAIN[idx_validate]))
		nnz.append( float((np.where(svc.coef_[0] != 0)[0]).shape[0]) )
	scores_mean.append(np.mean(scores))
	scores_std.append(np.std(scores))
	scores_CV_mean.append(np.mean(scores_CV))
	scores_CV_std.append(np.std(scores_CV))
	nnz_mean.append(np.mean(nnz))
	nnz_std.append(np.std(nnz))
	
	print("    C = %.2e, CV accuracy = %.2e +/- %.2e" % (C_val, scores_CV_mean[-1], scores_CV_std[-1]))

C_gsOpt = C_array[np.argmax(scores_CV_mean)]

scores_mean = np.array(scores_mean)
scores_std = np.array(scores_std)
scores_CV_mean = np.array(scores_CV_mean)
scores_CV_std = np.array(scores_CV_std)
nnz_mean = np.array(nnz_mean)
nnz_std = np.array(nnz_std)

plt.figure(1, figsize=(10, 8))
plt.clf()

ax = plt.subplot(111)
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

#plt.show()
plt.savefig('CValGridSearch.png', bbox_inches='tight')

print("  Best C value identified by grid search C_gsOpt = %e" % (C_gsOpt))
print("  Printing plot %s to file" % ("CValGridSearch.png"))
print("")


# - optimizing C value initializing from best grid search result
print("  (2) Optimizing C value initializing from best grid search result C = %e..." % (C_gsOpt))

svc.set_params(C=C_gsOpt)

print("     Performing optimization of log10(C) to tolerance in CV accuracy of %.2e..." % (optFtol)) 

optResult = scipy.optimize.minimize(objFun, math.log(C_gsOpt,10), args=(svc,crossVal,data_desc_TRAIN,data_resp_TRAIN), method='Powell', options={'ftol': optFtol, 'disp': False})
if optResult.success == False:
	print optResult
	print("ERROR: Optimization of C value failed; aborting.")
	sys.exit()

print("     Optimization complete")

C_opt = math.pow(10,optResult.x)

print("  Numerical optimization identified C_opt = %e " % (C_opt))
print("")




# - training classifier on all training data at optimal C value
print("  (3) Training classifier on all training data at optimal C = %e" % (C_opt))

svc.set_params(C=C_opt, probability=True)

svc = svc.fit(data_desc_TRAIN, data_resp_TRAIN)

with open("training_weights.csv",'w') as fout:

	for header in headers_desc:
		fout.write(header + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	
	for weight in svc.coef_[0]:
		fout.write(str(weight) + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")

print("   \-> Writing weights to %s" % ("training_weights.csv"))

joblib.dump(svc, 'svc.pkl')

print("   \-> Writing classifier to %s" % ("svc.pkl"))

print("")
print("")




# TESTING CLASSIFIER
print("  == TESTING CLASSIFIER on test data ==")
print("")


# - testing optimally trained classifier on test data
print("  (1) Testing optimally trained classifier...")
print("")

TEST_ACCURACY = svc.score(data_desc_TEST, data_resp_TEST)
data_resp_TEST_PREDSCORE = svc.decision_function(data_desc_TEST)

# - accuracy
print("    Accuracy = %.3f" % (TEST_ACCURACY))
print("")


# - ordering test instances by distance from the margin and estimating classification probabilities of each test instance
print("    Sorting test instances by distance to margin and estimated class membership probabilities")

idx_sort = np.argsort(data_resp_TEST_PREDSCORE)
idx_sort = idx_sort[::-1]

data_resp_TEST_CLASSPROB = svc.predict_proba(data_desc_TEST)

with open("sorted_test_instances.csv",'w') as fout:

	fout.write(headers_file + ",")
	fout.write(headers_index + ",")
	fout.write(headers_resp + ",")
	fout.write("prediction" + ",")
	fout.write("distToMargin" + ",")
	fout.write("P(-1)" + ",")
	fout.write("P(+1)" + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")

	for ii in idx_sort:
		fout.write(data_file_TEST[ii] + ",")
		fout.write(data_index_TEST[ii] + ",")
		fout.write(str(data_resp_TEST[ii]) + ",")
		if data_resp_TEST_PREDSCORE[ii] >= 0:
			fout.write(str(1) + ",")
		else:
			fout.write(str(-1) + ",")
		fout.write(str(data_resp_TEST_PREDSCORE[ii]) + ",")
		fout.write(str(data_resp_TEST_CLASSPROB[ii,0]) + ",")
		fout.write(str(data_resp_TEST_CLASSPROB[ii,1]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()
		fout.write("\n")

print("     \-> Writing sorted test instances to %s" % ("sorted_test_instances.csv"))
print("")


# - computing TP, TN, FP, FN and associated statistics as a function of threshold away from margin
#	Ref: Porto et al. PLOS ONE December 2012 | Volume 7 | Issue 12 | e51444
print("    Computing test metrics as a function of threshold")

nPoints=100
thresh_array_pos = np.linspace(float(nPoints/2+1)/float(nPoints/2)*np.max(data_resp_TEST_PREDSCORE),0,nPoints/2+1)
thresh_array_neg = np.linspace(0,float(nPoints/2+1)/float(nPoints/2)*np.min(data_resp_TEST_PREDSCORE),nPoints/2+1)
thresh_array = np.concatenate((thresh_array_pos, thresh_array_neg[1:]))

tp_array = np.empty([0,])
fp_array = np.empty([0,])
tn_array = np.empty([0,])
fn_array = np.empty([0,])
tpr_array = np.empty([0,])
tnr_array = np.empty([0,])
fpr_array = np.empty([0,])
acc_array = np.empty([0,])
ppv_array = np.empty([0,])
fdr_array = np.empty([0,])
npv_array = np.empty([0,])
for_array = np.empty([0,])
mcc_array = np.empty([0,])

for thresh in thresh_array:
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for ii in range(0,data_resp_TEST.size):
		if ( data_resp_TEST[ii] == +1 ):
			if ( data_resp_TEST_PREDSCORE[ii] >= thresh ):
				tp += 1
			else:
				fn += 1
		elif ( data_resp_TEST[ii] == -1 ):
			if ( data_resp_TEST_PREDSCORE[ii] >= thresh ):
				fp += 1
			else:
				tn += 1
		else:
			print("ERROR: data_resp_TEST[%d] not {-1,+1}" % (ii))
			sys.exit()
	
	tp_array = np.append(tp_array,tp)
	fp_array = np.append(fp_array,fp)
	tn_array = np.append(tn_array,tn)
	fn_array = np.append(fn_array,fn)
	tpr_array = np.append(tpr_array,float(tp)/float(tp+fn))
	tnr_array = np.append(tnr_array,float(tn)/float(tn+fp))
	fpr_array = np.append(fpr_array,float(fp)/float(tn+fp))
	acc_array = np.append(acc_array,float(tp+tn)/float(tp+tn+fn+fp))
	if (tp+fp) != 0:
		ppv_array = np.append(ppv_array,float(tp)/float(tp+fp))
		fdr_array = np.append(fdr_array,float(fp)/float(tp+fp))
	else:
		ppv_array = np.append(ppv_array,np.nan)
		fdr_array = np.append(fdr_array,np.nan)
	if (tn+fn) != 0:
		npv_array = np.append(npv_array,float(tn)/float(tn+fn))
		for_array = np.append(for_array,float(fn)/float(tn+fn))
	else:
		npv_array = np.append(npv_array,np.nan)
		for_array = np.append(for_array,np.nan)
	if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0:
		mcc_array = np.append(mcc_array,float((tp*tn)-(fp*fn))/np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
	else:
		mcc_array = np.append(mcc_array,np.nan)

# -- printing
with open("threshold_metrics.csv",'w') as fout:
	
	fout.write("threshold" + ",")
	fout.write("TP" + ",")
	fout.write("FP" + ",")
	fout.write("TN" + ",")
	fout.write("FN" + ",")
	fout.write("TPR = sensitivity = hit rate = recall" + ",")
	fout.write("TNR = specificity" + ",")
	fout.write("FPR = fallout = 1 - specificity" + ",")
	fout.write("accuracy" + ",")
	fout.write("positive predictive value (PPV) = precision" + ",")
	fout.write("false discovery rate (FDR) = 1 - PPV" + ",")
	fout.write("negative predictive value (NPV)" + ",")
	fout.write("false omission rate (FOR) = 1 - NPV" + ",")
	fout.write("Matthews correlation coefficient (MCC)" + ",")
	fout.seek(-1, os.SEEK_END)
	fout.truncate()	
	fout.write("\n")
	
	for ii in range(0,thresh_array.size):
		fout.write(str(thresh_array[ii]) + ",")
		fout.write(str(tp_array[ii]) + ",")
		fout.write(str(fp_array[ii]) + ",")
		fout.write(str(tn_array[ii]) + ",")
		fout.write(str(fn_array[ii]) + ",")
		fout.write(str(tpr_array[ii]) + ",")
		fout.write(str(tnr_array[ii]) + ",")
		fout.write(str(fpr_array[ii]) + ",")
		fout.write(str(acc_array[ii]) + ",")
		fout.write(str(ppv_array[ii]) + ",")
		fout.write(str(fdr_array[ii]) + ",")
		fout.write(str(npv_array[ii]) + ",")
		fout.write(str(for_array[ii]) + ",")
		fout.write(str(mcc_array[ii]) + ",")
		fout.seek(-1, os.SEEK_END)
		fout.truncate()	
		fout.write("\n")

print("     \-> Writing metrics to %s" % ("threshold_metrics.csv"))


# -- ROC
AUROC = np.trapz(tpr_array, fpr_array)

plt.figure(1, figsize=(10, 8))
plt.clf()

ax = plt.subplot(111)
plt.plot(fpr_array, tpr_array, 'b', label='ROC (area = %.3f)' % AUROC)
plt.plot([0, 1], [0, 1], 'k--', label='luck (area = %.3f)' % 0.50)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('FPR = FP/(FP+TN) = 1 - specificity')
plt.ylabel('TPR = TP/(TP+FN) = sensitivity')
plt.title('Receiver Operating Curve (accuracy = %.3f, AUROC = %.3f)' % (TEST_ACCURACY, AUROC))
plt.legend(loc='best')
#plt.show()
plt.savefig('ROC.png', bbox_inches='tight')

print("     \-> Printing ROC plot to %s (AUROC = %.3f)" % ("ROC.png",AUROC))


# -- precision-recall curve
plt.figure(1, figsize=(10, 8))
plt.clf()

ax = plt.subplot(111)
plt.plot(tpr_array, ppv_array, 'r', label='precision-recall')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('TPR = TP/(TP+FN) = sensitivity = recall')
plt.ylabel('PPV = TP/(TP+FP) = precision')
plt.title('Precision-Recall Plot')
#plt.legend(loc='best')
#plt.show()
plt.savefig('precision-recall.png', bbox_inches='tight')

print("     \-> Printing precision-recall plot to %s" % ("precision-recall.png"))
print("")


print("  DONE!")
print("")



## author: Saugat Bhattacharyya
## ErrP detection from a P300 dataset

from __future__ import division
import cPickle
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import sklearn.cross_validation as cv
import sklearn.metrics as metrics
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

print 'loading data'
train_feat = pd.read_csv('Across_Subject/train_feat.csv', header=None)
#print train_feat.shape

test_feat = pd.read_csv('Across_Subject/test_feat.csv',header=None)
#print test_feat.shape

Train = pd.read_csv('Across_Subject/train_meta.csv', header=None)
#print Train.shape

Test = pd.read_csv('Across_Subject/test_meta.csv',header=None)
#print Test.shape

Train=np.append(Train,train_feat, axis=1)
#print Train.shape
Test=np.append(Test,test_feat, axis=1)
#print Test.shape

print 'loading labels'
train_labels = pd.read_csv('Across_Subject/train_labels.csv', header=None)
print train_labels.shape

test_labels = pd.read_csv('Across_Subject/test_labels.csv', header=None)
print test_labels.shape

print 'Training...'

linda = LDA()
quada = QDA(reg_param=0.07)#
lgregl1 = LogisticRegression(penalty='l1',C=0.15)
lgregl2 = LogisticRegression(penalty='l2',C=0.15)

#CV=cv.KFold(len(train_labels),n_folds=10, shuffle = True) 

#for i,(train,test) in enumerate(CV):
#	out_linda=linda.fit(Train[train],train_labels.values[train].ravel()).predict_proba(Train[test])
#	out_quada=quada.fit(Train[train,:],train_labels.values[train].ravel()).predict_proba(Train[test])
#	out_lgregl1=lgregl1.fit(Train[train],train_labels.values[train].ravel()).predict_proba(Train[test])
#	out_lgregl2=lgregl2.fit(Train[train],train_labels.values[train].ravel()).predict_proba(Train[test])

#	cl_linda=linda.fit(Train[train],train_labels.values[train].ravel()).predict(Train[test])
#	cl_quada=quada.fit(Train[train],train_labels.values[train].ravel()).predict(Train[test])
#	cl_lgregl1=lgregl1.fit(Train[train],train_labels.values[train].ravel()).predict(Train[test])
#	cl_lgregl2=lgregl2.fit(Train[train],train_labels.values[train].ravel()).predict(Train[test])

#	acc_linda = metrics.accuracy_score(train_labels.values[test].ravel(), cl_linda)
#	acc_quada = metrics.accuracy_score(train_labels.values[test].ravel(), cl_quada)
#	acc_lgregl1 = metrics.accuracy_score(train_labels.values[test].ravel(), cl_lgregl1)
#	acc_lgregl2 = metrics.accuracy_score(train_labels.values[test].ravel(), cl_lgregl2)
#	print 'The individual 10-CV accuracy is', acc_linda, acc_quada, acc_lgregl1, acc_lgregl2

#	f_linda = metrics.f1_score(train_labels.values[test].ravel(), cl_linda)
#	f_quada = metrics.f1_score(train_labels.values[test].ravel(), cl_quada)
#	f_lgregl1 = metrics.f1_score(train_labels.values[test].ravel(), cl_lgregl1)
#	f_lgregl2 = metrics.f1_score(train_labels.values[test].ravel(), cl_lgregl2)
#	print 'The individual 10-CV f1-score is', f_linda, f_quada, f_lgregl1, f_lgregl2

#	fpr_linda, tpr_linda, thresholds = metrics.roc_curve(train_labels.values[test].ravel(), out_linda[:,1], pos_label=1)
#	fpr_quada, tpr_quada, thresholds = metrics.roc_curve(train_labels.values[test].ravel(), out_quada[:,1], pos_label=1)
#	fpr_lgregl1, tpr_lgregl1, thresholds = metrics.roc_curve(train_labels.values[test].ravel(), out_lgregl1[:,1], pos_label=1)
#	fpr_lgregl2, tpr_lgregl2, thresholds = metrics.roc_curve(train_labels.values[test].ravel(), out_lgregl2[:,1], pos_label=1)

#	print 'The individual 10-CV FPR is', fpr_linda, fpr_lgregl1, fpr_lgregl2, thresholds
#	print 'The individual 10-CV TPR is', tpr_linda, tpr_lgregl1, tpr_lgregl2, thresholds

#	auc_linda = metrics.auc(fpr_linda,tpr_linda)
#	auc_quada = metrics.auc(fpr_quada,tpr_quada)
#	auc_lgregl1 = metrics.auc(fpr_lgregl1,tpr_lgregl1)
#	auc_lgregl2 = metrics.auc(fpr_lgregl2,tpr_lgregl2)
#	print 'The individual 10-CV AUC is', auc_linda, auc_quada, auc_lgregl1, auc_lgregl2

#	out=(out_linda+out_quada+out_lgregl1+out_lgregl2)/4
#	cl= np.around((cl_linda+cl_quada+cl_lgregl1+cl_lgregl2)/4)

#	acc = metrics.accuracy_score(train_labels.values[test].ravel(), cl)
#	print 'CV-Accuracy of ensemble is',acc

#	f = metrics.f1_score(train_labels.values[test].ravel(), cl)
#	print 'The ensemble CV-f1-score is', f

#	fpr, tpr, thresholds = metrics.roc_curve(train_labels.values[test].ravel(), out[:,1], pos_label=1)
#	print 'The FPR of CV-ensemble is', fpr, thresholds
#	print 'The TPR of CV-ensemble is', tpr, thresholds

#	auc = metrics.auc(fpr,tpr)
#	print 'CV-AUC of ensemble is',auc

#print 'Self-Test'
#st_linda=linda.fit(Train.values,train_labels.values[:,1].ravel()).predict_proba(Train.values)
#st_quada1=quada.fit(Train.values,train_labels.values[:,1].ravel()).predict_proba(Train.values)
#st_lgregl1=lgregl1.fit(Train.values,train_labels.values[:,1].ravel()).predict_proba(Train.values)
#st_lgregl2=lgregl2.fit(Train.values,train_labels.values[:,1].ravel()).predict_proba(Train.values)

#st=(st_linda+st_quada+st_lgregl1+st_lgregl2)/4
#fpr, tpr, thresholds = metrics.roc_curve(train_labels.values[:,1].ravel(), st[:,1],pos_label=1)
#auc_st = metrics.auc(fpr,tpr)
#print 'AUC of ensemble is',auc_st

print 'Using Test dataset'

pred_linda=linda.fit(Train,train_labels.values.ravel()).predict_proba(Test)
pred_quada=quada.fit(Train,train_labels.values.ravel()).predict_proba(Test)
pred_lgregl1=lgregl1.fit(Train,train_labels.values.ravel()).predict_proba(Test)
pred_lgregl2=lgregl2.fit(Train,train_labels.values.ravel()).predict_proba(Test)

class_linda=linda.fit(Train,train_labels.values.ravel()).predict(Test)
class_quada=quada.fit(Train,train_labels.values.ravel()).predict(Test)
class_lgregl1=lgregl1.fit(Train,train_labels.values.ravel()).predict(Test)
class_lgregl2=lgregl2.fit(Train,train_labels.values.ravel()).predict(Test)

preds=(pred_linda+pred_lgregl1+pred_lgregl2)/4
preds = preds[:,1]

cls = np.around((class_linda+class_quada+class_lgregl1+class_lgregl2)/4)

acc = metrics.accuracy_score(test_labels.values.ravel(), cls)
print 'Test accuracy of ensemble is',acc

f = metrics.f1_score(test_labels.values.ravel(), cls)
print 'The ensemble test f1-score is', f

fpr, tpr, thresholds = metrics.roc_curve(test_labels.values.ravel(), preds, pos_label=1)

auc = metrics.auc(fpr,tpr)
print 'Test AUC of ensemble is',auc

print 'Done'


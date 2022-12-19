#encoding:utf-8

import numpy as np
import math
from MARK55_BN_global import MARK55_BN_global
from MARK55_BN import MARK55_BN
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as metrics_auc

def MARK56_XGBDIM_validation_beta2(
		X_global, X_local, label, Nv, W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global, M_local, Beta, Gamma, Sigma, W_local, N_model,lr_model,N_epoch):

	h = np.zeros((Nv, 1))
	# [C, T, _] = np.shape(X_global)
	N_sp = np.shape(W_global)[0]
	N_te = np.shape(Q_global)[1]

	X_global_BN = MARK55_BN_global(X_global, Gamma_global, Beta_global, M_global, Sigma_global)
	for n in range(Nv):
		for m in range(N_sp):
			for p in range(N_te):
				h[n, 0] = h[n, 0] + W_global[m, :] @ X_global_BN[:, :, n] @ Q_global[:, p]/N_sp/N_te
		h[n, 0] = h[n, 0] + b_global
		h[n, 0] = 0.3 * h[n, 0]
	if N_model > 1:
		T_local = np.shape(X_local)[1]
		for k in range(N_model - 1):
			X_local_BN_k = MARK55_BN(X_local[:, :, k], Gamma[k], Beta[k], M_local[k, :], Sigma[k, :])
			for n in range(Nv):
				f = W_local[k, 1: T_local+1] @ X_local_BN_k[n, :].T + W_local[k, 0]
				h[n, 0] = h[n, 0] + lr_model[k] * f
	s = 1 / (1 + np.exp(-1 * h))

	idx_1 = np.where(np.array(label) == 1)
	idx_2 = np.where(np.array(label) == 0)

	y_predicted_final = s.copy()
	y_predicted_final[np.where(y_predicted_final >= 0.5)] = int(1)
	y_predicted_final[np.where(y_predicted_final < 0.5)] = int(0)

	acc = np.sum((y_predicted_final == label) != 0) / np.shape(y_predicted_final)[0]

	n_positive = np.sum(label == 1)
	n_negative = np.sum(label == 0)

	n_tp = np.sum(y_predicted_final.T * label.T)
	n_fp = np.sum((y_predicted_final.T == 1)*(label.T == 0))


	tpr = n_tp / n_positive
	fpr = n_fp / n_negative
	fpr_1, tpr_1, thresholds = roc_curve(label, s)
	auc = metrics_auc(fpr_1, tpr_1)

	return acc, tpr, fpr, auc



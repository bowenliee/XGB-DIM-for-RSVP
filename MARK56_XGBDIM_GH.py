#encoding:utf-8
import numpy as np
from MARK55_BN_global import MARK55_BN_global
from MARK55_BN import MARK55_BN
import math

def MARK56_XGBDIM_GH(X_global, X_local, label, C1, C0, W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,
					 W_local, M_local, Beta, Gamma, Sigma, N_model, lr_model, N_epoch):
	Ch, Te, Ns = np.shape(X_global)

	h = np.zeros((Ns, 1))
	N_sp = np.shape(W_global)[0]
	N_te = np.shape(Q_global)[1]


	X_global_BN_1 = MARK55_BN_global(X_global, Gamma_global, Beta_global, M_global, Sigma_global)
	for n in range(Ns):
		for m in range(N_sp):
			for p in range(N_te):
				h[n, 0] = h[n, 0] + W_global[m, :] @ X_global_BN_1[:, :, n] @ Q_global[:, p]/N_sp/N_te
		h[n, 0] = h[n, 0] + b_global
		h[n, 0] = h[n, 0] * 0.3
	if N_model > 1:
		T_local = np.shape(X_local)[1]
		for k in range(N_model-1): # N_model is number of model in the last iteration, Number of loc model is N_model-1

			X_local_BN_k = MARK55_BN(X_local[:, :, k], Gamma[k], Beta[k], M_local[k, :], Sigma[k, :])
			for n in range(Ns):
				f = W_local[k, 1: T_local+1] @ X_local_BN_k[n, :].T + W_local[k, 0]
				# print(f)
				h[n, 0] = h[n, 0] + lr_model[k] * f

	s_k_1 = 1 / (1 + np.exp(-1 * h))
	'G_k(n)'
	G_k = (C1 - C0) * s_k_1 * label - C1 * label + C0 * s_k_1
	# G_k = np.zeros((N, 1))
	# for n in range(N):
	# 	G_k[n] = (C1-C0) * s_k_1[n, 0] * label[n, 0] - C1 * label[n, 0] + C0 * s_k_1[n, 0]

	'H_k(n)'
	H_k = ((C1 - C0) * label + C0) * s_k_1 * (1 - s_k_1)
	# H_k = np.zeros((N, 1))
	# for n in range(N):
	# 	H_k[n] = ((C1 - C0) * label[n, 0] + C0) * s_k_1[n, 0] * (1 - s_k_1[n, 0])
	# print('G and H done!')
	return G_k, H_k
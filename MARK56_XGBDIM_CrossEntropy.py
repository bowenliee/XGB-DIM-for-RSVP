#encoding:utf-8
import numpy as np
from MARK55_BN_global import MARK55_BN_global
from MARK55_BN import MARK55_BN
import math


def MARK56_XGBDIM_CrossEntropy(X_global,X_local,label,\
                        W_global,Q_global,b_global,Gamma_global,Beta_global,Sigma_global,M_global,\
                        W_local,M_local,u,v,Sigma,Beta,Gamma,N_epoch,lr_model):

	Ns = np.shape(X_local)[0]
	T_local = np.shape(X_local)[1]
	N_conv = np.shape(W_local)[0]
	N_model = N_conv + 1
	X_minibatch_BN_global = MARK55_BN_global(X_global, Gamma_global, Beta_global, M_global, Sigma_global)
	X_minibatch_BN = np.zeros((Ns, T_local, N_conv))
	for idx_conv in range(N_conv):
		X_minibatch_BN[:, :, idx_conv] = MARK55_BN(X_local[:, :, idx_conv], Gamma[idx_conv], Beta[idx_conv],
												   M_local[idx_conv, :], Sigma[idx_conv, :])
	Crossentropy = 0
	#s =np.empty((N,1))

	N_sp = np.shape(W_global)[0]
	N_te = np.shape(Q_global)[1]

	for n in range(Ns):
		h_k = 0
		for m in range(N_sp):
			for p in range(N_te):
				h_k = h_k + W_global[m,:] @ X_minibatch_BN_global[:,:,n] @ Q_global[:,p]/N_sp/N_te
		h_k = h_k + b_global
		h_k = 0.3 * h_k
		for k in range(N_conv):
			f_k = W_local[k,:] @ np.concatenate(([1],X_minibatch_BN[n,:,k])).T
			h_k = h_k + lr_model[k] * f_k

		s = 1 / (1 + np.exp(-1 * h_k))
		Crossentropy = Crossentropy - label[n] * np.log(s) / Ns - (1 - label[n]) * np.log(1 - s) / Ns

	return Crossentropy
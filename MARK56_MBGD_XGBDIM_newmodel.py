#encoding:utf-8
import numpy as np
import math
from MARK55_slidemean import MARK55_slidemean
from MARK55_BN_global import MARK55_BN_global
from MARK55_BN import MARK55_BN
from MARK55_MBGD_ADAM import MARK55_MBGD_ADAM
from MARK56_XGBDIM_validation_beta2 import MARK56_XGBDIM_validation_beta2

def MARK56_MBGD_XGBDIM_newmodel(X_train_global,X_validation_global,X_train,X_validation,label_train,label_validation,G_k,H_k,alpha,eta,lr_model,C1,C0,N_iteration,N_batch,N_model,N_epoch,\
    W_global,Q_global,b_global,Gamma_global, Beta_global,Sigma_global,M_global,W_local,Gamma,Beta,Sigma,M_local,u,v,mW,vW,mGamma,vGamma,mBeta,vBeta,mu,vu,mv,vv,\
    idx_batch,idx_model,idx_sub):

	beta1 = 0.9
	beta2 = 0.999
	it = idx_batch
	#idx_conv = idx_model - 1
	idx_conv = idx_model - 1
	'DATA parameter analysis '
	N_batch = np.shape(X_train)[0]
	[Nv, T_local, _] = np.shape(X_validation)
	N_win = idx_conv
	'Batch normalization // 1. mean and sigma for whold batch ! # NOT mini batch !'
	idx_BN = idx_batch
	m_old = M_local[idx_conv - 1, :]
	sigma_old = Sigma[idx_conv - 1, :]
	m_now = np.mean(X_train[:, :, idx_conv - 1], axis=0) #batch mean
	M_local[idx_conv - 1, :] = MARK55_slidemean(m_old, m_now, idx_BN * N_batch, N_batch)  # update mean
	# if idx_BN == 0:
	# 	M[idx_conv - 1, :] = MARK55_slidemean(m_old, m_now, (idx_BN ) * N, N)  # update mean
	# else:
	# 	M[idx_conv - 1, :] = MARK55_slidemean(m_old, m_now, (idx_BN - 1) * N, N) #update mean
	sigma_now = np.zeros(T_local)

	for n in range(N_batch):
		sigma_now = sigma_now + (X_train[n, :, idx_conv-1] - M_local[idx_conv-1, :])**2
	sigma_now = sigma_now / N_batch # batch sigma
	Sigma[idx_conv - 1, :] = MARK55_slidemean(sigma_old, sigma_now, idx_BN * N_batch, N_batch)  # update sigma

	# if idx_BN == 0:
	# 	Sigma[idx_conv - 1, :] = MARK55_slidemean(sigma_old, sigma_now, (idx_BN ) * N, N)  # update sigma
	# else:
	# 	Sigma[idx_conv-1, :] = MARK55_slidemean(sigma_old, sigma_now, (idx_BN - 1) * N, N) # update sigma

	sigma = Sigma[idx_conv-1, :]
	sigma[np.where(sigma == 0)] = np.mean(sigma)
	Sigma[idx_conv-1, :] = sigma.copy()

	'Iteration and update'
	Crossentropy = 0 # zeros(1, N_iteration);
	Accvalidation = 0
	mStepSize = 0
	stdStepSize = 0

	f_k = np.zeros((N_batch, 1))
	'Iteration'
	delta_w = 2 * eta * W_local[idx_conv-1, :]
	delta_w[0] = 0
	delta_gamma = 0
	delta_beta = 0
	delta_u = 0
	delta_v = 0
	X_minibatch_BN = np.zeros((N_batch, T_local, idx_conv))
	X_minibatch = X_train[:, :, idx_conv-1]

	X_minibatch_BN_global = MARK55_BN_global(X_train_global, Gamma_global, Beta_global, M_global, Sigma_global)
	for k in range(idx_conv):
		X_minibatch_BN[:, :, k] = MARK55_BN(X_train[:, :, k], Gamma[k], Beta[k], M_local[k, :], Sigma[k, :])
	for n in range(N_batch):
		f_k[n, 0] = W_local[idx_conv-1, 1: T_local+1] @ X_minibatch_BN[n, :, idx_conv-1].T + W_local[idx_conv-1, 0]
	# sum(W[idx_conv - 1, :] * np.concatenate([[1], X_minibatch_BN[n, :, idx_conv - 1]]))
	for n in range(N_batch):
		temp = G_k[n, 0] + H_k[n, 0] * f_k[n, 0]
		if not np.isnan(temp) and np.abs(temp) < float('inf'):
			for t in range(T_local):
				delta_w[t+1] = delta_w[t+1] + 1/N_batch * temp * X_minibatch_BN[n, t, idx_conv-1]
				delta_beta = delta_beta + 1/N_batch * temp * W_local[idx_conv - 1, t+1]
				delta_gamma = delta_gamma + 1 / N_batch * temp * W_local[idx_conv - 1, t + 1] * (
							X_minibatch[n, t] - M_local[idx_conv - 1, t]) / np.sqrt(Sigma[idx_conv - 1, t])

			delta_w[0] = delta_w[0] + temp/N_batch
		'Crossentropy'
		N_sp = np.shape(W_global)[0]
		N_te = np.shape(Q_global)[1]
		f_sigma = 0
		for m in range(N_sp):
			for p in range(N_te):
				f_sigma = f_sigma + W_global[m, :] @ X_minibatch_BN_global[:, :, n] @ Q_global[:, p]/N_sp/N_te
		f_sigma = f_sigma + b_global
		f_sigma = f_sigma * 0.3
		for k in range(idx_conv):
			f = W_local[k, 1:T_local+1] @ X_minibatch_BN[n, :, k].T + W_local[k, 0]
			f_sigma = f_sigma + lr_model[k] * f
		h_k = f_sigma.copy()
		s = 1/(1 + np.exp(-1 * h_k))
		Crossentropy = Crossentropy - label_train[n, 0] * np.log(s) / N_batch - (1 - label_train[n, 0]) * np.log(
			1 - s) / N_batch
		'ADAbound'
##test
	# res_tmp = MARK55_MBGD_ADAM(W_local[idx_conv - 1, :], delta_w,
	# 																					 mW[idx_conv - 1, :],
	# 																					 vW[idx_conv - 1, :], alpha,
	# 																					 beta1, beta2, it)
	# W_local[idx_conv - 1, :], mW[idx_conv - 1, :], vW[idx_conv - 1, :], lrw = res_tmp[0], res_tmp[1], res_tmp[2], res_tmp[3]

	W_local[idx_conv - 1, :], mW[idx_conv - 1, :], vW[idx_conv - 1, :], lrw = MARK55_MBGD_ADAM(W_local[idx_conv - 1, :], delta_w,
																						 mW[idx_conv - 1, :],
																						 vW[idx_conv - 1, :], alpha,
																						 beta1, beta2, it)

	Beta[idx_conv - 1], mBeta[idx_conv - 1], vBeta[idx_conv - 1], lrbeta = MARK55_MBGD_ADAM(Beta[idx_conv - 1],
																							delta_beta,
																							mBeta[idx_conv - 1],
																							vBeta[idx_conv - 1], alpha,
																							beta1, beta2, it)

	Gamma[idx_conv - 1], mGamma[idx_conv - 1], vGamma[idx_conv - 1], lrgamma = MARK55_MBGD_ADAM(Gamma[idx_conv - 1],
																								delta_gamma,
																								mGamma[idx_conv - 1],
																								vGamma[idx_conv - 1],
																								alpha, beta1, beta2, it)
	lr = lrw.copy()  # 存疑  lr = lrw[:]
	mStepSize = np.mean(lr)
	stdStepSize = np.std(lr)

	# print('Crossentropy = %f' % Crossentropy)
	'Validation'
	# print('Subject %d Model %d/300 idx_interation %d' % (idx_sub, idx_model, idx_BN))
	if (idx_model % 10) == 0 and idx_batch == N_iteration - 1:
		Accvalidation, tpr, fpr, auc = MARK56_XGBDIM_validation_beta2(X_validation_global, X_validation,
																	  label_validation, Nv, W_global, Q_global,
																	  b_global, Gamma_global, Beta_global, Sigma_global,
																	  M_global, M_local, Beta, Gamma, Sigma, W_local, idx_model,
																	  lr_model, N_epoch)

		# print(Accvalidation)
		# print(tpr)
		# print(fpr)
		# print(auc)
	else:
		Accvalidation = 0
		tpr = 0
		fpr = 0
		auc = 0

	return (
		W_local, Gamma, Beta, Sigma, M_local, u, v, mW, vW, mGamma, vGamma, mBeta, vBeta, mu, vu, mv, vv,
		Crossentropy,
		Accvalidation,
		tpr,
		fpr,
		auc,
		mStepSize, stdStepSize
	)






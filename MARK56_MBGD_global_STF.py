#encoding:utf-8
import numpy as np
import scipy.io as scio
import os
import h5py
import matplotlib.pyplot as plt
import math
from MARK55_slidemean import  MARK55_slidemean
from MARK55_BN_global import MARK55_BN_global
from MARK55_MBGD_ADAM import MARK55_MBGD_ADAM
from MARK56_XGBDIM_validation_beta2 import MARK56_XGBDIM_validation_beta2

def MARK56_MBGD_global_STF(X_train_global, X_validation_global, label_train, label_validation, C1, C0, alpha, eta,
						   lr_model, N_iteration, N_batch, N_model, N_epoch, W_global, Q_global, b_global,
						   Gamma_global, Beta_global, Sigma_global, M_global, mW_global, vW_global, mQ_global,
						   vQ_global, mb_global, vb_global, mGamma_global, vGamma_global, mBeta_global, vBeta_global,
						   idx_batch, idx_model, idx_sub):

	beta1 = 0.9
	beta2 = 0.999
	it = idx_batch
	'''DATA parameter analysis'''
	Ns = np.shape(X_train_global)[2]
	[Ch, Te, Nv] = np.shape(X_validation_global)
	'''Batch normalization // 1. mean and sigma for whole batch ! # NOT mini batch ! '''
	idx_BN = idx_batch
	m_old = M_global.copy()
	sigma_old = Sigma_global.copy()

	m_now = np.mean((X_train_global), axis=2) #batch mean  #check

	M_global = MARK55_slidemean(m_old, m_now, idx_BN * Ns, Ns)

	sigma_now = np.zeros((Ch, Te))
	for n in range(Ns):
		sigma_now = sigma_now + (X_train_global[:, :, n] - M_global) * (X_train_global[:, :, n] - M_global)
	sigma_now = sigma_now / Ns # batch sigma
	Sigma_global = MARK55_slidemean(sigma_old, sigma_now, idx_BN * Ns, Ns) # update sigma
	# sigma = Sigma_global
	#sigma(sigma == 0) = np.mean(np.mean(sigma, 1))
	# Sigma_global = sigma

	'''
	Iteration and update
	'''

	############## Initialization
	Crossentropy = 0 # np.zeros((1, N_iteration));
	# Accvalidation = Crossentropy
	# mStepSize = Crossentropy
	# stdStepSize = Crossentropy
##################  Iteration

	delta_W_global = 2 * eta * W_global
	delta_b_global = 0
	delta_Q_global = 2 * eta * Q_global #w: (1 + T) * C
	delta_Gamma_global = 0
	delta_Beta_global = 0

	N_sp = np.shape(W_global)[0]
	N_te = np.shape(Q_global)[1]

	X_minibatch_global_BN = np.zeros((Ch, Te, Ns))
	X_minibatch = X_train_global.copy()
	X_minibatch_global_BN = MARK55_BN_global(X_train_global, Gamma_global, Beta_global, M_global, Sigma_global)
	'prediction before update'

	f = np.zeros((Ns, 1))
	for n in range(N_batch):
		for m in range(N_sp):
			for p in range(N_te):
				f[n, 0] = f[n, 0] + np.dot(np.dot(W_global[m, :].reshape((1, -1)), X_minibatch_global_BN[:, :, n]), (Q_global[:, p])/N_sp/N_te)
		f[n, 0] = f[n, 0] + b_global
	h = f.copy()
	s = 1 / (1 + np.exp(-1 * h))

	for n in range(N_batch):
		temp = (C1 - C0) * s[n, 0] * label_train[n, 0] - C1 * label_train[n, 0] + C0 * s[n, 0]
		Rn = np.zeros((Ch, 1))
		for p in range(N_te):
			Rn = Rn + np.dot(X_minibatch_global_BN[:, :, n], Q_global[:, p]).reshape((60, 1))
		Un = np.zeros(Te)
		for m in range(N_sp):
			Un = Un + np.dot(W_global[m, :], X_minibatch_global_BN[:, :, n])

		X_n_BN0 = (X_minibatch[:, :, n] - M_global) / np.sqrt(Sigma_global)

		if not np.isnan(temp) and abs(temp) < float('inf'):
			for m in range(N_sp):
				for c in range(Ch):
					delta_W_global[m, c] = delta_W_global[m, c] + 1 / N_batch * temp * Rn[c] / N_sp / N_te
				for p in range(N_te):
					delta_Gamma_global = delta_Gamma_global + 1 / N_batch * temp * W_global[m, :] @ X_n_BN0 @ Q_global[:, p].reshape((-1, 1))/N_sp/N_te
					delta_Beta_global = delta_Beta_global + 1 / N_batch * temp * W_global[m, :] @  np.ones((Ch, Te)) @ Q_global[:, p].reshape((-1, 1))/N_sp/N_te
			for p in range(N_te):
				for t in range(Te):
					delta_Q_global[t, p] = delta_Q_global[t, p] + 1/N_batch * temp * Un[t]/N_sp/N_te
			delta_b_global = delta_b_global + temp / N_batch

		'Crossentropy'
		Crossentropy = Crossentropy - label_train[n, 0] * np.log(s[n, 0]) / N_batch - (1 - label_train[n, 0]) * np.log(1 - s[n, 0]) / N_batch

	W_global, mW_global, vW_global, lrw = MARK55_MBGD_ADAM(W_global, delta_W_global, mW_global, vW_global, alpha, beta1,
														   beta2, it)

	Q_global, mQ_global, vQ_global, lrq = MARK55_MBGD_ADAM(Q_global, delta_Q_global, mQ_global, vQ_global, alpha, beta1,
														   beta2, it)

	b_global, mb_global, vb_global, lrb = MARK55_MBGD_ADAM(b_global, delta_b_global, mb_global, vb_global, alpha, beta1,
														   beta2, it)

	Beta_global, mBeta_global, vBeta_global, lrbeta = MARK55_MBGD_ADAM(Beta_global, delta_Beta_global, mBeta_global,
																	   vBeta_global, alpha, beta1, beta2, it)

	Gamma_global, mGamma_global, vGamma_global, lrgamma = MARK55_MBGD_ADAM(Gamma_global, delta_Gamma_global,
																			 mGamma_global, vGamma_global, alpha, beta1,
																			 beta2, it)
	lr = lrw.copy()
	mStepSize = np.mean(lr)
	stdStepSize = np.std(lr)

	# print('Crossentropy = %f' % Crossentropy)

	'Validation'
	# print('Subject %d Model %d/300 idx_interation %d' % (idx_sub, idx_model, idx_batch))
	if (idx_model % 10 == 0 and idx_batch == N_iteration - 1) or (idx_model == 1 and idx_batch == N_iteration - 1):
		Accvalidation, tpr, fpr, auc = MARK56_XGBDIM_validation_beta2(X_validation_global, 0, label_validation, Nv,
																	  W_global, Q_global, b_global, Gamma_global,
																	  Beta_global, Sigma_global, M_global, 0, 0, 0, 0,
																	  0, idx_model, lr_model, N_epoch)

		# print(Accvalidation)
		# print(tpr)
		# print(fpr)
		# print(auc)
	else:
		Accvalidation = 0
		tpr = 0
		fpr = 0
		auc = 0
	# print(['Subject ', num2str(idx_sub), ' Model ', num2str(idx_model), '/', num2str(N_model + 1), ' idx_BN ',
	# 	  num2str(idx_BN), ', ', ' 鈻?', repmat('鈻?', 1, floor(idx_batch / N_iteration * 40)),
	# 	  repmat('||', 1, 40 - floor(idx_batch / N_iteration * 40)), ' ' num2str(floor(100 * idx_batch / N_iteration)),
	# 	  '%', ' ', running{mod(idx_batch, 4) + 1}])
	return (
		W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,
		mW_global, vW_global, mQ_global, vQ_global, mb_global, vb_global,
		mGamma_global, vGamma_global, mBeta_global, vBeta_global,
		Crossentropy,
		Accvalidation,
		tpr,
		fpr,
		auc,
		mStepSize, stdStepSize
	)




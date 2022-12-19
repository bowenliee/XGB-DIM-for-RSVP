# encoding:utf-8
import numpy as np
import scipy.io as scio
import os
import h5py
import matplotlib.pyplot as plt
import random
from MARK56_MBGD_global_STF import MARK56_MBGD_global_STF
from MARK56_XGBDIM_GH import MARK56_XGBDIM_GH
from MARK56_MBGD_XGBDIM_newmodel import MARK56_MBGD_XGBDIM_newmodel
from MARK56_XGBDIM_CrossEntropy import MARK56_XGBDIM_CrossEntropy
import time


def MARK56_train_XGBDIM(trainset, validationset, channel, channel_loc, sub_idx, window_st, \
						win_len, window_ov, channel_conv, chan_len, eta, alpha, lr_model, Nb, \
						N_iteration, N_epoch, C1, C0, max_N_model):
	start_time = time.time()
	T_local = win_len * chan_len
	N_win = np.shape(window_st)[0]
	N_chanwin = np.shape(channel_conv)[0]
	N_conv = N_win * N_chanwin
	N_model = min([N_epoch * N_conv, max_N_model])
	N_batch = Nb * 2

	'''Training and validation set'''
	# Training set
	print('Import Training data')
	N_trainsets = np.shape(trainset)[0]#1  # np.shape(trainset)
	All_X1 = []
	All_X2 = []
	for idx_set in range(N_trainsets):
		# Sub1_L1
		# data = h5py.File(os.path.join(r'D:\sub1data\sub1', 'sub'+str(sub_idx)+'_'+str(trainset[idx_set])+'_data.mat'))  #因为N_trainsets只有一个，所以先写死测试一下；
		data = h5py.File(
			os.path.join(r'D:\RSVP_data', 'Sub' + str(sub_idx) + '_' + str(trainset[idx_set]) + '.mat'))
		X1 = np.transpose(data['X1'])
		X2 = np.transpose(data['X2'])
		if All_X1 == []:
			All_X1 = X1.copy()
			All_X2 = X2.copy()
		else:
			All_X1 = np.concatenate((All_X1, X1), axis=2)
			All_X2 = np.concatenate((All_X2, X2), axis=2)
	X1 = All_X1.copy()
	del All_X1
	X2 = All_X2.copy()
	del All_X2
	K1 = np.shape(X1)[2]
	K2 = np.shape(X2)[2]
	X1 = np.delete(X1, list(range(50)), axis=1)
	X2 = np.delete(X2, list(range(50)), axis=1)
	for k in range(K1):
		X1[:, :, k] = X1[:, :, k] - np.mean(X1[:, :, k], axis=1)[:, np.newaxis]  # 拓展到二维才能用numpy的广播机制；
	for k in range(K2):
		X2[:, :, k] = X2[:, :, k] - np.mean(X2[:, :, k], axis=1)[:, np.newaxis]
	plt.figure()
	plt.plot(np.arange(np.shape(X1[:, window_st[0]:window_ov[N_win -1], :])[1]),
			 np.mean(X1[channel[31]-1, window_st[0]:window_ov[N_win -1], :], 1))

	plt.plot(np.arange(np.shape(X1[:, window_st[0]:window_ov[N_win -1], :])[1]),
			 np.mean(X2[channel[31]-1, window_st[0]:window_ov[N_win -1], :], 1))

	print('Import Training data finished')

	Tset_train = np.zeros((K1, T_local, N_chanwin * N_win))
	NTset_train = np.zeros((K2, T_local, N_chanwin * N_win))
	idx_conv = -1  #每次内循环自加1，python索引从0开始，保证最后索引不超出；
	for idx_chan in range(N_chanwin):
		for idx_win in range(N_win):
			idx_conv += 1
			for k in range(K1):
				cup = np.squeeze(X1[channel[channel_conv[idx_chan, :] -1]-1, window_st[idx_win] - 1:window_ov[idx_win], k])  #注意索引要减1
				Tset_train[k, :, idx_conv] = cup.T.flatten()
			for k in range(K2):
				cup = np.squeeze(X2[channel[channel_conv[idx_chan, :] -1]-1, window_st[idx_win] - 1:window_ov[idx_win], k])  #注意索引要减1
				NTset_train[k, :, idx_conv] = cup.T.flatten()

	'restruction for global model'
	Tset_train_global = X1.copy()
	NTset_train_global = X2.copy()

	del X1
	del X2
	'''
	Validation sets
	'''


	print('Import Test data')
	N_validationsets = np.shape(validationset)[0]
	All_X1 = []
	All_X2 = []
	for idx_set in range(N_validationsets):
		# data = h5py.File(os.path.join(r'D:\sub1data\sub1', 'sub'+str(sub_idx)+'_'+str(validationset[idx_set])+'_data.mat'))  #因为N_trainsets只有一个，所以先写死测试一下；
		data = h5py.File(
			os.path.join(r'D:\RSVP_data', 'Sub' + str(sub_idx) + '_' + str(trainset[idx_set]) + '.mat'))

		X1 = np.transpose(data['X1'])
		X2 = np.transpose(data['X2'])
		if All_X1 == []:
			All_X1 = X1.copy()
			All_X2 = X2.copy()
		else:
			All_X1 = np.concatenate((All_X1, X1), axis=2)
			All_X2 = np.concatenate((All_X2, X2), axis=2)


	#data_vd = h5py.File(os.path.join(r'D:\科研\XGBDIM' , 'sub1_'+str(validationset[0])+'_data.mat'))


	X1 = All_X1.copy()
	del All_X1
	X2 = All_X2.copy()
	del All_X2

	K1v = np.shape(X1)[2]
	K2v = np.shape(X2)[2]
	X1 = np.delete(X1, list(range(50)), axis=1)
	X2 = np.delete(X2, list(range(50)), axis=1)
	for k in range(K1v):
		X1[:,:,k] = X1[:,:,k] - np.mean(X1[:,:,k],1)[:, np.newaxis]
	for k in range(K2v):
		X2[:,:,k] = X2[:,:,k] - np.mean(X2[:,:,k],1)[:, np.newaxis]

	plt.plot(np.arange(np.shape(X1[:,window_st[0]:window_ov[N_win-1],:])[1]),np.mean(X1[channel[31]-1,window_st[0]:window_ov[N_win-1],:],1))
	plt.plot(np.arange(np.shape(X2[:,window_st[0]:window_ov[N_win-1],:])[1]),np.mean(X2[channel[31]-1,window_st[0]:window_ov[N_win-1],:],1))


	print('Data import finished')
	'''
	 restruction for local model boosting
	'''
	Tset_validation = np.zeros((K1v, T_local, N_chanwin * N_win))
	NTset_validation = np.zeros((K2v, T_local, N_chanwin * N_win))
	idx_conv = -1
	for idx_chan in range(N_chanwin):
		for idx_win in range(N_win):
			idx_conv = idx_conv + 1
			for k in range(K1v):
				cup = np.squeeze(X1[channel[channel_conv[idx_chan, :] -1 ]-1, window_st[idx_win] - 1:window_ov[idx_win] , k])  #注意索引要减1
				Tset_validation[k, :, idx_conv] = cup.T.flatten()
			for k in range(K2v):
				cup = np.squeeze(X2[channel[channel_conv[idx_chan, :] -1 ]-1, window_st[idx_win] - 1:window_ov[idx_win] , k])  #注意索引要减1
				NTset_validation[k, :, idx_conv] = cup.T.flatten()

	'''
	restruction for global model
	'''
	Tset_validation_global = X1.copy()
	del X1
	NTset_validation_global = X2.copy()
	del X2

	X_validation = np.concatenate((Tset_validation,NTset_validation),axis=0)
	X_validation_global = np.concatenate((Tset_validation_global[:,:,:],NTset_validation_global[:,:,:]),axis=2)

	Nb = np.min([Nb, K1])
	label_validation = np.concatenate((np.ones((K1v,1)),np.zeros((K2v,1))),0)


	'''adapative learning rate for boosting'''
	if N_model > 1:
		M1 = np.mean(Tset_train,0).T
		M2 = np.mean(NTset_train,0).T
		distance_0_1 = np.zeros(N_conv)
		for idx_conv in range(N_conv):
			R1 = 0
			for k in range(K1):
				R1 = R1 + np.matrix(Tset_train[k, :, idx_conv] - M1[idx_conv, :]).T * np.matrix(Tset_train[k, :, idx_conv] - M1[idx_conv, :])
			R1 = R1 / K1
			R2 = 0
			for k in range(K2):
				R2 = R2 + np.matrix(NTset_train[k, :, idx_conv] - M2[idx_conv, :]).T * np.matrix(NTset_train[k, :, idx_conv] - M2[idx_conv, :])
			R2 = R2 / K2
			distance_0_1[idx_conv] = np.trace(np.linalg.pinv(R1+R2) * np.matrix(M2[idx_conv, :] - M1[idx_conv, :]).T * np.matrix(M2[idx_conv, :] - M1[idx_conv, :]))
		dist = np.sort(distance_0_1)[::-1]
		I_sort = np.argsort(distance_0_1)[::-1]
		# print(I_sort)
		dist = dist/max(dist)

		weight_conv = -1 / (N_conv - 1) / (N_conv - 1) * np.multiply(np.arange(N_conv), np.arange(N_conv)) + 1
		lr_model = 0.5 * np.multiply(weight_conv, dist) # weight_conv =
		# lr_model = np.array([])
		# for idx_conv in range(N_conv):
		# 	lr_model = np.append(lr_model, np.ones((1, N_epoch)) * weight_conv[idx_conv])

		NTset_train = NTset_train[:, :, I_sort[:N_model]]
		Tset_train = Tset_train[:, :, I_sort[:N_model]]
		X_validation = X_validation[:, :, I_sort[:N_model]]

	'Downsampling negative samples'
	idx_nontarget_train = [i for i in range(K2)]
	random.shuffle(idx_nontarget_train)
	idx_selected_2 = idx_nontarget_train[:K1]
	# idx_selected_2 = np.arange(0, K2, round(K2/K1))

	NTset_train = NTset_train[idx_selected_2, :, :]
	NTset_train_global = NTset_train_global[:, :, idx_selected_2]
	K2 = np.shape(idx_selected_2)[0]

	# Nb = min((K1, K2))
	N_batch = Nb * 2
	label_train = np.concatenate((np.ones((Nb, 1)), np.zeros((Nb, 1))), 0)
	label_all = np.concatenate((np.ones((K1, 1)), np.zeros((K2, 1))), 0)


	boost_clock = time.time()
	print('boosting prepared...., %.2f' % (boost_clock - start_time))
	'''INITIALIZATION'''
	N_sp = 1
	N_te = 1

	W_global = 0.01 + 0.01 * np.random.rand(N_sp, np.shape(Tset_train_global)[0])
	Q_global = 0.01 + 0.01 * np.random.rand(np.shape(Tset_train_global)[1], N_te)
	b_global = 0
	Gamma_global = 1
	Beta_global = 0
	Sigma_global = np.zeros((np.shape(Tset_train_global)[0], np.shape(Tset_train_global)[1]))
	M_global = np.zeros((np.shape(Tset_train_global)[0], np.shape(Tset_train_global)[1]))

	mW_global = W_global * 0
	vW_global = mW_global.copy()

	mQ_global = Q_global * 0
	vQ_global = mQ_global.copy()

	mb_global = 0
	vb_global = 0

	mBeta_global = 0
	vBeta_global = 0

	mGamma_global = 0
	vGamma_global = 0

	'local model'

	W_local = np.zeros((N_model, T_local+1)) #0.01 + 0.01 * np.random.rand(N_model, T_local + 1)
	#W_local[:, 0] = W_local[:, 0] * 0

	M_local = np.zeros((N_model, T_local))  # continue from the same window
	Sigma = np.zeros((N_model, T_local)) #continue from the same window
	Gamma = np.ones((N_model, 1))
	Beta = np.zeros((N_model, 1))
	u = 1
	v = 0
	mW = W_local * 0
	vW = mW.copy()
	mBeta = Beta.copy()
	vBeta = Beta.copy()
	mGamma = Beta.copy()
	vGamma = Beta.copy()
	mu = 0
	vu = 0
	mv = 0
	vv = 0
	init_clock = time.time()

	'''
	XGB-DIM
	'''
	idx_model = 0
	# for idx_conv in range(N_model+1):
	# 	idx_model += 1
	# 	print(idx_conv)
	# 	print(idx_model)
	#
	for idx_conv in range(N_model+1):
		# for idx_epoch in range(N_epoch):
		if idx_conv == 0:
			# Crossentropy_all = []
			# Accvalidation_all = np.zeros(N_iteration)
			# tpr_all = np.zeros(N_iteration)
			# fpr_all = np.zeros(N_iteration)
			# auc_all = np.zeros(N_iteration)
			idx_model += 1 #
			# Crossentropy = np.empty([N_iteration]) * 0
			# Accvalidation = Crossentropy
			# tpr = Crossentropy
			# fpr = Crossentropy
			# auc = Crossentropy

			for idx_iteration in range(N_iteration):

				idx_sample_target_train = [i for i in range(K1)]
				random.shuffle(idx_sample_target_train)
				select_1_idx = idx_sample_target_train[:Nb]

				idx_sample_nontarget_train = [i for i in range(K2)]
				random.shuffle(idx_sample_nontarget_train)
				select_2_idx = idx_sample_nontarget_train[:Nb]

				# select_1_idx = np.arange(Nb)
				# select_2_idx = np.arange(Nb)

				# select_1_idx = np.random.permutation(K1)[:Nb]
				# select_2_idx = np.random.permutation(K2)[:Nb]
				# select_1_idx = np.arange(Nb)
				# select_1_idx = np.arange(Nb)
				X_train_global = np.concatenate((Tset_train_global[:, :, select_1_idx], NTset_train_global[:, :, select_2_idx]), 2)

				W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,mW_global, vW_global, mQ_global, vQ_global, mb_global, vb_global, mGamma_global, vGamma_global,mBeta_global, vBeta_global, \
				 Crossentropy, Accvalidation, tpr, fpr, auc, mStepSize, stdStepSize\
				 =MARK56_MBGD_global_STF(X_train_global, X_validation_global, label_train, label_validation, C1, C0,alpha, eta, lr_model, N_iteration, N_batch, N_model, N_epoch,\
				W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,mW_global, vW_global, mQ_global, vQ_global, mb_global, vb_global, mGamma_global, vGamma_global, mBeta_global, vBeta_global,\
				idx_iteration, idx_model, sub_idx)

				Crossentropy_all = Crossentropy.copy()
				Accvalidation_all = Accvalidation
				tpr_all = tpr
				fpr_all = fpr
				auc_all = auc

			print('Subject %d Model %d/300 done !' % (sub_idx, idx_model))
			print('ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))

		else: #
			G_k_T, H_k_T = MARK56_XGBDIM_GH(Tset_train_global, Tset_train[:, :, :idx_conv], np.ones((np.shape(Tset_train)[0], 1)), C1, C0,\
			W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global, W_local[:idx_conv, :], M_local[:idx_conv, :], \
			Beta[:idx_conv], Gamma[:idx_conv], Sigma[:idx_conv], idx_model, lr_model, N_epoch)

			G_k_N, H_k_N = MARK56_XGBDIM_GH(NTset_train_global, NTset_train[:, :, :idx_conv], np.zeros((np.shape(NTset_train)[0], 1)), C1, C0, \
			W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global, W_local[:idx_conv, :], M_local[:idx_conv, :], \
			Beta[:idx_conv], Gamma[:idx_conv], Sigma[:idx_conv], idx_model, lr_model, N_epoch)

			idx_model = idx_model + 1

			mu = 0
			vu = 0
			mv = 0
			vv = 0
			Crossentropy = np.zeros(N_iteration)
			Accvalidation = Crossentropy.copy()
			tpr = Crossentropy.copy()
			fpr = Crossentropy.copy()
			auc = Crossentropy.copy()
			# M[idx_model, :] = M[idx_model - 1, :]
			# Sigma[idx_model, :] = Sigma[idx_model - 1, :]
			# M = np.append(M,M[idx_model - 2,:].reshape((1,-1)),axis = 0)
			# Sigma = np.append(Sigma,Sigma[idx_model - 2,:].reshape((1,-1)),axis = 0)###每一个子模型的更新都伴随着对sigma增加一行，sigma是每个模型所对应数据的标准差；

			for idx_iteration in range(N_iteration):

				idx_sample_target_train = [i for i in range(K1)]
				random.shuffle(idx_sample_target_train)
				select_1_idx = idx_sample_target_train[:Nb]

				idx_sample_nontarget_train = [i for i in range(K2)]
				random.shuffle(idx_sample_nontarget_train)
				select_2_idx = idx_sample_nontarget_train[:Nb]

				# select_1_idx = np.arange(Nb)
				# select_2_idx = np.arange(Nb)

				X_train_global = np.concatenate([Tset_train_global[:, :, select_1_idx], NTset_train_global[:, :, select_2_idx]], axis=2)
				X_train = np.concatenate((Tset_train[select_1_idx, :, :idx_conv], NTset_train[select_2_idx, :, :idx_conv]), axis=0)
				G_k = np.concatenate((G_k_T[select_1_idx], G_k_N[select_2_idx]), axis=0)
				H_k = np.concatenate((H_k_T[select_1_idx], H_k_N[select_2_idx]), axis=0)

				# if idx_model == 10 and idx_iteration == 19:
				# 	print('stop')
				# W, Gamma, Beta, Sigma, M, u, v, mW, vW, mGamma, vGamma, mBeta, vBeta, mu, vu, mv, vv, Crossentropy, Accvalidation, tpr, fpr, auc, mStepSize, stdStepSize
				W_local[:idx_conv, :], Gamma[:idx_conv], Beta[:idx_conv], Sigma[:idx_conv, :], M_local[:idx_conv, :], u, v,\
					mW[:idx_conv, :], vW[:idx_conv, :], mGamma[:idx_conv], vGamma[:idx_conv], mBeta[:idx_conv],\
					vBeta[:idx_conv], mu, vu, mv, vv, Crossentropy[idx_iteration], Accvalidation[idx_iteration], tpr[idx_iteration], fpr[idx_iteration], auc[idx_iteration],\
					mStepSize, stdStepSize = MARK56_MBGD_XGBDIM_newmodel(X_train_global, X_validation_global, X_train, X_validation, label_train,\
					label_validation, G_k, H_k, alpha, eta, lr_model, C1, C0, N_iteration, N_batch, N_model, N_epoch, \
					W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,\
					W_local[:idx_conv, :], Gamma[:idx_conv], Beta[:idx_conv], Sigma[:idx_conv, :], M_local[:idx_conv, :], u, v,\
					mW[:idx_conv, :], vW[:idx_conv, :], mGamma[:idx_conv], vGamma[:idx_conv], mBeta[:idx_conv],\
					vBeta[:idx_conv], mu, vu, mv, vv, idx_iteration, idx_model, sub_idx)


			# print('ACC %f TPR %f FPR %f AUC %f' % (Accvalidation[idx_iteration], tpr[idx_iteration], fpr[idx_iteration], auc[idx_iteration]))

				# W, Gamma, Beta, Sigma, M, u, v, mW, vW, mGamma, vGamma, mBeta, vBeta, mu, vu, mv, vv, Crossentropy, Accvalidation, tpr, fpr, auc, mStepSize, stdStepSize ==MARK56_MBGD_XGBDIM_newmodel(X_train_global, X_validation_global, X_train, X_validation, label_train,\
				# label_validation, G_k, H_k, alpha, eta, lr_model, C1, C0, N_iteration, N_batch,N_model, N_epoch, W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,\
				# W[ : idx_conv,:], Gamma[ : idx_conv], Beta[ : idx_conv], Sigma[ : idx_conv,:], M[ : idx_conv,:], u, v, mW[ : idx_conv,:], vW[ : idx_conv,:], mGamma[ : idx_conv], vGamma[ : idx_conv], \
				# mBeta[: idx_conv], vBeta[ : idx_conv], mu, vu, mv, vv, idx_iteration, idx_model, sub_idx)
				#
				# [W[:idx_conv, :], Gamma[: idx_conv], Beta[: idx_conv], Sigma[: idx_conv, :], M[: idx_conv, :], u, v, \
				# mW[: idx_conv, :], vW[: idx_conv, :], mGamma[: idx_conv], vGamma[: idx_conv], mBeta[: idx_conv], \
				# vBeta[: idx_conv], mu, vu, mv, vv, Crossentropy[idx_iteration], Accvalidation[idx_iteration], tpr[
				# 	idx_iteration], fpr[idx_iteration], \
				# auc[idx_iteration], mStepSize, stdStepSize]=[W, Gamma, Beta, Sigma, M, u, v, mW, vW, mGamma, vGamma, mBeta, vBeta, mu, vu, mv, vv, Crossentropy, Accvalidation, tpr, fpr, auc, mStepSize, stdStepSize]
				# Crossentropy_all[idx_iteration] = Crossentropy
				# Accvalidation_all[idx_iteration] = Accvalidation
				# tpr_all[idx_iteration] = tpr
				# fpr_all[idx_iteration] = fpr
				# auc_all[idx_iteration] = auc
			print('Subject %d Model %d/300 done !' % (sub_idx, idx_model))

		if idx_model % 10 == 0 and idx_model != 0:

			print('ACC %f TPR %f FPR %f AUC %f' % (Accvalidation[N_iteration - 1], tpr[N_iteration - 1], fpr[N_iteration - 1], auc[N_iteration - 1]))
			Crossentropy_all=np.append(Crossentropy_all, \
			np.ones(10) * MARK56_XGBDIM_CrossEntropy\
			(np.concatenate((Tset_train_global,NTset_train_global),axis=2),np.concatenate((Tset_train[:,:,:idx_conv],NTset_train[:,:,:idx_conv]),axis=0),label_all,\
			W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,W_local[:idx_conv,:],M_local[:idx_conv,:],u,v,Sigma[:idx_conv,:],Beta[:idx_conv,:],Gamma[:idx_conv,:],N_epoch,lr_model))



			Accvalidation_all =np.append(Accvalidation_all,np.ones((1, 10)) * Accvalidation[N_iteration - 1])
			tpr_all = np.append(tpr_all, np.ones((1, 10)) * tpr[N_iteration - 1])
			fpr_all = np.append(fpr_all, np.ones((1, 10)) * fpr[N_iteration - 1])
			auc_all = np.append(auc_all, np.ones((1, 10)) * auc[N_iteration - 1])



			# tpr_all[idx_model - 10: idx_model] = np.ones((1, 10)) * tpr[N_iteration]
			# fpr_all[idx_model - 10: idx_model] = np.ones((1, 10)) * fpr[N_iteration]
			# auc_all[idx_model - 10: idx_model] = np.ones((1, 10)) * auc[N_iteration]
			# Crossentropy_all[idx_model -10 : idx_model] = np.ones(10) * MARK56_XGBDIM_CrossEntropy(np.concatenate((Tset_train_global,NTset_train_global),axis=2),np.concatenate((Tset_train[:,:,:idx_conv],NTset_train[:,:,:idx_conv]),axis=0),label_all,\
			# W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global,W[:idx_conv,:],M[:idx_conv,:],u,v,Sigma[:idx_conv,:],Beta[:idx_conv,:],Gamma[:idx_conv,:],N_epoch,lr_model)
			# Accvalidation_all[idx_model - 10: idx_model] = np.ones((1, 10)) * Accvalidation[N_iteration]
			# tpr_all[idx_model - 10: idx_model] = np.ones((1, 10)) * tpr[N_iteration]
			# fpr_all[idx_model - 10: idx_model] = np.ones((1, 10)) * fpr[N_iteration]
			# auc_all[idx_model - 10: idx_model] = np.ones((1, 10)) * auc[N_iteration]
			# if idx_model > 1:
	# plt.figure()
	# plt.plot(np.arange(301), Crossentropy_all)
	# plt.figure()
	# plt.plot(np.arange(301), Accvalidation_all)
	# plt.figure()
	# plt.plot(np.arange(301), tpr_all)
	# plt.figure()
	# plt.plot(np.arange(301), fpr_all)
	# plt.figure()
	# plt.plot(np.arange(301), auc_all)
	# plt.show()
	print('Model Training Finished !')
	# X_validation = np.concatenate(Tset_validation(:,:,:), NTset_validation(:,:,:))
	# X_validation_global(:,:,:) = cat(3, Tset_validation_global(:,:,:), NTset_validation_global(:,:,:));




	return W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global, W_local, Gamma, Beta, Sigma, M_local, u, v, lr_model, I_sort, idx_model, Crossentropy_all, Accvalidation_all, tpr_all, fpr_all, auc_all
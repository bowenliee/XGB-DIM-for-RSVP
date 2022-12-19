#encoding:utf-8
import numpy as np
import os
from MARK56_train_XGBDIM import MARK56_train_XGBDIM

sub_train = np.array(range(1, 65)) #1 : 40; %40
temp = np.array(range(1, 55))
channel_loc = np.array([temp[0:9], temp[9:18], temp[18:27], temp[27:36], temp[36:45], temp[45:54]])

Nx_channel = np.shape(channel_loc)[1]
chan_xlen = 3
step = 3
channel_xst = np.arange(0, Nx_channel - chan_xlen + 1)[0::step]

Ny_channel = np.shape(channel_loc)[0]
chan_ylen = 3
step = 3
channel_yst = np.arange(0, Ny_channel - chan_ylen + 1)[0::step]

chan_len = chan_xlen * chan_ylen
idx_conv = 0

channel_conv = []
for idx_y in channel_yst:
	for idx_x in channel_xst:
		idx_conv = idx_conv + 1
		cup = channel_loc[idx_y: idx_y + chan_ylen, idx_x: idx_x + chan_xlen]
		channel_conv.append(cup.T.reshape(-1))

channel_conv = np.array(channel_conv)
channel = np.array([i for k in [list(range(6, 54)), [58, 54, 60], list(range(55, 58))] for i in k])

win_len = 6
step = 3   #round(win_len/2)
window_st = np.array(range(1, 250 - win_len + 1, step))
window_ov = window_st + win_len - 1

N_model = np.shape(window_st)[0] * np.shape(channel_conv)[0]

 # T = win_len*size(channel,2) 
lr_model = 0.1
trainset = np.array([1]) #['1']
validationset = np.array([2,3,4]) #['2', '3', '4']
 # parameters for MBGD
alpha = 0.05  # initial learning rate
eta = 0.5  # L2 regularization coefficient
N_iteration = 20   # number of iterations
N_epoch = 1 
Nb = 32   # number of  postive or negative trials in one mini-batch
C1 = 1 
C0 = 1


"""Learning the XGB-DIM models"""
for sub_idx in sub_train:

	max_N_model = 299

	W_global, Q_global, b_global, Gamma_global, Beta_global, Sigma_global, M_global, W_local, Gamma, Beta, Sigma, M_local, u, v, lr_model, conv_sort, N_model, Crossentropy_all, Accvalidation_all, tpr_all, fpr_all, auc_all = MARK56_train_XGBDIM(
		trainset, validationset, channel, channel_loc, sub_idx, window_st, win_len, window_ov, channel_conv, chan_len,
		eta, alpha, 0, Nb, N_iteration, N_epoch, C1, C0, max_N_model)


	model_path = 'D:/XGBDIM/PY_SingleShort'


	filename = 'Model_' + str(sub_idx) + '.npz'
	np.savez(os.path.join(model_path, filename), W_global=W_global, Q_global=Q_global, b_global=b_global,
			Gamma_global=Gamma_global, Beta_global=Beta_global, Sigma_global=Sigma_global, M_global=M_global,
			W_local=W_local, Gamma=Gamma, Beta=Beta, Sigma=Sigma, M_local=M_local, lr_model=lr_model,
			conv_sort=conv_sort, Accvalidation_all=Accvalidation_all, tpr_all=tpr_all, fpr_all=fpr_all, auc_all=auc_all)
	# c = np.load(os.path.join(model_path, filename))
	# c['a']



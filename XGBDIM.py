'''
XGBDIM.py
Author: Bowen Li, Tsinghua University
'''
import numpy as np
import os
import h5py
# import time
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as metrics_auc
'''
data_path: the path of the data
sub_idx: the index of the subject
trainset: the index of the training set, example: np.array([1,2,3])
validationset: the index of the validation set, example: np.array([4,5,6])
model_path: the path of the model
n_cutpoint: the number of cutpoints, the removed points are ones before img onset, example: 50 (200 ms for 250 Hz)
win_len: the length of the window
chan_xlen: the number of the channels in x direction
chan_ylen: the number of the channels in y direction
step_x: the step of the channels in x direction
step_y: the step of the channels in y direction
eta: the regulation coefficient of the L2 norm
alpha: learning rate
Nb: the number of P/N samples in one batch
N_iteration: the number of iterations
C1: the weight of the positive samples
C0: the weight of the negative samples
max_N_model: the maximum number of sub models
gstf_weight: the weight of the GSTF, example: 0.3
validation_flag: whether to use the validation set
validation_step: the step of the validation, example: 100, means that the validation is performed every 100 sub models
crossentropy_flag: whether to calculate the cross entropy loss

'''
class XGBDIM():

    def __init__(self, data_path, sub_idx, trainset, validationset,
                 model_path,
                 n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
                 eta, alpha, Nb, N_iteration, C1, C0, max_N_model, gstf_weight,
                 validation_flag, validation_step, crossentropy_flag):

        self.data_path = data_path
        self.model_path = model_path

        self.trainset = trainset
        self.validationset = validationset
        self.sub_idx = sub_idx

        self.win_len = win_len
        self.chan_xlen = chan_xlen
        self.chan_ylen = chan_ylen
        self.chan_len = chan_xlen * chan_ylen
        self.step_x = step_x
        self.step_y = step_y
        self.T_local = win_len * self.chan_len

        self.eta = eta
        self.alpha = alpha
        self.Nb = Nb
        self.N_iteration = N_iteration
        self.C1 = C1
        self.C0 = C0
        self.max_N_model = max_N_model
        self.gstf_weight = gstf_weight
        self.validation_flag = validation_flag
        self.validation_step = validation_step
        self.crossentropy_flag = crossentropy_flag

        temp = np.array(range(1, 55))
        self.channel_loc = np.array([temp[0:9], temp[9:18], temp[18:27], temp[27:36], temp[36:45], temp[45:54]])

        self.n_cutpoint = n_cutpoint # number of points before stimulus
        self.beta1 = 0.9
        self.beta2 = 0.999

    def get_3Dconv(self):

        Nx_channel = np.shape(self.channel_loc)[1]
        channel_xst = np.arange(0, Nx_channel - self.chan_xlen + 1)[0::self.step_x]

        Ny_channel = np.shape(self.channel_loc)[0]
        channel_yst = np.arange(0, Ny_channel - self.chan_ylen + 1)[0::self.step_y]

        self.chan_len = self.chan_xlen * self.chan_ylen

        idx_conv = 0
        channel_conv = []
        for idx_y in channel_yst:
            for idx_x in channel_xst:
                idx_conv = idx_conv + 1
                cup = self.channel_loc[idx_y: idx_y + self.chan_ylen, idx_x: idx_x + self.chan_xlen]
                channel_conv.append(cup.T.reshape(-1))

        self.channel_conv = np.array(channel_conv)
        self.channel = np.array(list(range(6, 54)) + [58, 54, 60] + list(range(55, 58)))

        step_win = int(self.win_len/2)# round(win_len/2)
        self.window_st = np.array(range(1, 250 - self.win_len + 1, step_win))
        self.window_ov = self.window_st + self.win_len - 1

        self.N_model = np.shape(self.window_st)[0] * np.shape(self.channel_conv)[0]
        self.N_win = np.shape(self.window_st)[0]
        self.N_chanwin = np.shape(self.channel_conv)[0]
        self.N_conv = self.N_win * self.N_chanwin
        self.N_model = min([self.N_conv, self.max_N_model])

    def read_data(self, dataset):
        if dataset is not None:
            print('Importing data')
            N_trainsets = np.shape(dataset)[0]  # 1  # np.shape(trainset)
            All_X1 = []
            All_X2 = []
            for idx_set in range(N_trainsets):
                # Sub1_L1
                # data = h5py.File(os.path.join(r'D:\sub1data\sub1', 'sub'+str(sub_idx)+'_'+str(trainset[idx_set])+'_data.mat'))  #因为N_trainsets只有一个，所以先写死测试一下；
                data = h5py.File(
                    os.path.join(
                        self.data_path, 'sub' + str(self.sub_idx) + '_' + str(dataset[idx_set]) + '_data.mat'
                    )
                )
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
            print('Importing data finished')
            return X1, X2, K1, K2
        else:
            print('Dataset is None')
            exit(1)

    def get_3D_cuboids(self, X1, X2, K1, K2):
        Tset = np.zeros((K1, self.T_local, self.N_chanwin * self.N_win))
        NTset = np.zeros((K2, self.T_local, self.N_chanwin * self.N_win))
        idx_conv = -1
        for idx_chan in range(self.N_chanwin):
            for idx_win in range(self.N_win):
                idx_conv += 1
                for k in range(K1):
                    cup = np.squeeze(
                        X1[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
                        self.window_st[idx_win] - 1:self.window_ov[idx_win],
                        k])  # 注意索引要减1
                    Tset[k, :, idx_conv] = cup.T.flatten()
                for k in range(K2):
                    cup = np.squeeze(
                        X2[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
                        self.window_st[idx_win] - 1:self.window_ov[idx_win],
                        k])  # 注意索引要减1
                    NTset[k, :, idx_conv] = cup.T.flatten()

        return Tset, NTset

    def preprocess(self, X1, X2, K1, K2):
        X1 = np.delete(X1, list(range(self.n_cutpoint)), axis=1)
        X2 = np.delete(X2, list(range(self.n_cutpoint)), axis=1)
        for k in range(K1):
            X1[:, :, k] = X1[:, :, k] - np.mean(X1[:, :, k], axis=1)[:, np.newaxis]
        for k in range(K2):
            X2[:, :, k] = X2[:, :, k] - np.mean(X2[:, :, k], axis=1)[:, np.newaxis]

        return X1, X2

    def get_data(self, dataset):
        X1, X2, K1, K2 = self.read_data(dataset)
        X1, X2 = self.preprocess(X1, X2, K1, K2)
        Tset, NTset = self.get_3D_cuboids(X1, X2, K1, K2)
        Tset_global = X1.copy()
        NTset_global = X2.copy()

        return Tset, NTset, Tset_global, NTset_global, K1, K2

    def get_order_step(self):
        '''adapative learning rate for boosting'''
        print('Learning the order and step of sub models')
        if self.N_model > 1:
            M1 = np.mean(self.Tset_train, 0).T
            M2 = np.mean(self.NTset_train, 0).T
            distance_0_1 = np.zeros(self.N_conv)
            for idx_conv in range(self.N_conv):
                R1 = 0
                for k in range(self.K1):
                    R1 = R1 + np.matrix(self.Tset_train[k, :, idx_conv] - M1[idx_conv, :]).T * np.matrix(
                        self.Tset_train[k, :, idx_conv] - M1[idx_conv, :])
                R1 = R1 / self.K1
                R2 = 0
                for k in range(self.K2):
                    R2 = R2 + np.matrix(self.NTset_train[k, :, idx_conv] - M2[idx_conv, :]).T * np.matrix(
                        self.NTset_train[k, :, idx_conv] - M2[idx_conv, :])
                R2 = R2 / self.K2
                distance_0_1[idx_conv] = np.trace(
                    np.linalg.pinv(R1 + R2) * np.matrix(M2[idx_conv, :] - M1[idx_conv, :]).T * np.matrix(
                        M2[idx_conv, :] - M1[idx_conv, :]))
            dist = np.sort(distance_0_1)[::-1]
            self.I_sort = np.argsort(distance_0_1)[::-1]

            # print(I_sort)
            dist = dist / max(dist)
            weight_conv = -1 / (self.N_conv - 1) / (self.N_conv - 1) * np.multiply(np.arange(self.N_conv),
                                                                                   np.arange(self.N_conv)) + 1
            self.lr_model = 0.5 * np.multiply(weight_conv, dist)
            print('Learning the order and step of sub models finished!')
            # return I_sort, lr_model
        else:
            print('Please run the global model! This is not a ensemble model!')
            exit(1)

    def batchnormalize(self, X_local, gamma, beta, m, sigma):
        Ns = np.shape(X_local)[0]
        X_local_BN = gamma * (X_local - np.ones((Ns, 1)) * m) / np.sqrt(np.ones((Ns, 1)) * sigma) + beta
        return X_local_BN

    def batchnormalize_global(self, X_global, Gamma_global, Beta_global, M_global, Sigma_global):
        [Ch, Te, Ns] = np.shape(X_global)
        X_global_BN = np.empty((Ch, Te, Ns)) * np.nan
        for n in range(Ns):
            X_global_BN[:, :, n] = Gamma_global * (X_global[:, :, n] - M_global) / np.sqrt(
                Sigma_global) + Beta_global
        return X_global_BN

    def slidemean(self, x_old, x_now, n_old, n_now):
        x_update = (x_old * n_old + x_now * n_now) / (n_old + n_now)
        return x_update

    def ADAM(self, x, deltax, mx, vx):
        mx = self.beta1 * mx + (1 - self.beta1) * deltax
        vx = self.beta2 * vx + (1 - self.beta2) * (deltax * deltax)

        beta1_t = 1 / 9
        beta2_t = 1 / 999
        m_ = mx / (1 - beta1_t)
        v_ = vx / (1 - beta2_t)

        solution = x - self.alpha / np.sqrt(v_ + self.alpha) * m_
        x = solution.copy()
        lrx = np.min(self.alpha / np.sqrt(v_ + self.alpha))
        return x, mx, vx, lrx

    def decision_value(self, X_minibatch_BN, X_minibatch_BN_global, N_model, Ns):
        h = np.zeros((Ns, 1))
        N_sp = np.shape(self.W_global)[0]
        N_te = np.shape(self.Q_global)[1]

        for n in range(Ns):
            for m in range(N_sp):
                for p in range(N_te):
                    h[n, 0] = h[n, 0] + self.W_global[m, :] @ X_minibatch_BN_global[:, :, n] @ self.Q_global[:, p] / N_sp / N_te
            h[n, 0] = h[n, 0] + self.b_global
            h[n, 0] = self.gstf_weight * h[n, 0]
            for k in range(N_model-1):
                f_k = self.W_local[k, 1:self.T_local+1] @ X_minibatch_BN[n, :, k].T + self.W_local[k, 0]
                h[n, 0] = h[n, 0] + self.lr_model[k] * f_k

        s = 1 / (1 + np.exp(-1 * h))
        return s, h

    def CrossEntropy(self, X_global, X_local, label, N_model):

        Ns = np.shape(X_local)[0]
        X_minibatch_BN_global = self.batchnormalize_global(X_global, self.Gamma_global, self.Beta_global, self.M_global, self.Sigma_global)
        X_minibatch_BN = np.zeros((Ns, self.T_local, N_model-1))
        for idx_conv in range(N_model-1):
            X_minibatch_BN[:, :, idx_conv] = self.batchnormalize(X_local[:, :, idx_conv], self.Gamma[idx_conv], self.Beta[idx_conv],
                                                       self.M_local[idx_conv, :], self.Sigma[idx_conv, :])
        Crossentropy = 0
        # s =np.empty((N,1))
        s, h = self.decision_value(X_minibatch_BN, X_minibatch_BN_global, N_model, Ns)
        for n in range(Ns):
            Crossentropy = Crossentropy - label[n] * np.log(s[n, 0]) / Ns - (1 - label[n]) * np.log(1 - s[n, 0]) / Ns

        return Crossentropy

    def validation(self, N_model):

        Ns = np.shape(self.X_validation_global)[2]
        X_global_BN = self.batchnormalize_global(self.X_validation_global, self.Gamma_global, self.Beta_global, self.M_global, self.Sigma_global)
        X_minibatch_BN = np.zeros((Ns, self.T_local, N_model - 1))
        for idx_conv in range(N_model - 1):
            X_minibatch_BN[:, :, idx_conv] = self.batchnormalize(self.X_validation[:, :, idx_conv], self.Gamma[idx_conv],
                                                                 self.Beta[idx_conv],
                                                                 self.M_local[idx_conv, :], self.Sigma[idx_conv, :])

        s, h = self.decision_value(X_minibatch_BN, X_global_BN, N_model, Ns)

        idx_1 = np.where(np.array(self.label_validation) == 1)
        idx_2 = np.where(np.array(self.label_validation) == 0)

        y_predicted_final = s.copy()
        y_predicted_final[np.where(y_predicted_final >= 0.5)] = int(1)
        y_predicted_final[np.where(y_predicted_final < 0.5)] = int(0)

        acc = np.sum((y_predicted_final == self.label_validation) != 0) / np.shape(y_predicted_final)[0]

        n_positive = np.sum(self.label_validation == 1)
        n_negative = np.sum(self.label_validation == 0)
        n_tp = np.sum(y_predicted_final.T * self.label_validation.T)
        n_fp = np.sum((y_predicted_final.T == 1) * (self.label_validation.T == 0))

        tpr = n_tp / n_positive
        fpr = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(self.label_validation, s)
        auc = metrics_auc(fpr_1, tpr_1)
        return acc, tpr, fpr, auc

    def get_GH(self, X_global, X_local, label, N_model):
        Ch, Te, Ns = np.shape(X_global)
        X_global_BN_1 = self.batchnormalize_global(X_global, self.Gamma_global, self.Beta_global, self.M_global, self.Sigma_global)
        # X_minibatch_BN = np.zeros((Ns, self.T_local, N_model - 1))
        if N_model == 1:
            s, self.h_GH_temp = self.decision_value(0, X_global_BN_1, N_model, Ns)
        else:
            for n in range(Ns):
                X_minibatch_BN = self.batchnormalize(X_local[n, :, N_model-2].reshape((1, self.T_local)), self.Gamma[N_model-2],
                                                                     self.Beta[N_model-2],
                                                                     self.M_local[N_model-2, :], self.Sigma[N_model-2, :])
                f_k = self.W_local[N_model-2, 1:self.T_local + 1] @ X_minibatch_BN.T + self.W_local[N_model-2, 0]
                # self.W_local[k, 1:self.T_local + 1] @ X_minibatch_BN[n, :, k].T + self.W_local[k, 0]
                self.h_GH_temp[n, 0] = self.h_GH_temp[n, 0] + self.lr_model[N_model-2] * f_k

        s = 1 / (1 + np.exp(-self.h_GH_temp))
        G_k = (self.C1 - self.C0) * s * label - self.C1 * label + self.C0 * s
        H_k = ((self.C1 - self.C0) * label + self.C0) * s * (1 - s)
        return G_k, H_k

    def MBGD_global_STF(self, idx_batch, idx_model):

        Ns = np.shape(self.X_train_global)[2]
        [Ch, Te, Nv] = np.shape(self.X_validation_global)
        '''Batch normalization // 1. mean and sigma for whole batch ! # NOT mini batch ! '''
        idx_BN = idx_batch
        m_old = self.M_global.copy()
        sigma_old = self.Sigma_global.copy()

        m_now = np.mean((self.X_train_global), axis=2) #batch mean  #check

        self.M_global = self.slidemean(m_old, m_now, idx_BN * Ns, Ns)

        sigma_now = np.zeros((Ch, Te))
        for n in range(Ns):
            sigma_now = sigma_now + (self.X_train_global[:, :, n] - self.M_global) * (self.X_train_global[:, :, n] - self.M_global)
        sigma_now = sigma_now / Ns # batch sigma
        self.Sigma_global = self.slidemean(sigma_old, sigma_now, idx_BN * Ns, Ns) # update sigma

        '''
        Iteration and update
        '''
        Crossentropy = 0 # np.zeros((1, N_iteration));

        delta_W_global = 2 * self.eta * self.W_global
        delta_b_global = 0
        delta_Q_global = 2 * self.eta * self.Q_global #w: (1 + T) * C
        delta_Gamma_global = 0
        delta_Beta_global = 0

        N_sp = np.shape(self.W_global)[0]
        N_te = np.shape(self.Q_global)[1]

        X_minibatch_global_BN = np.zeros((Ch, Te, Ns))
        X_minibatch = self.X_train_global.copy()
        X_minibatch_global_BN = self.batchnormalize_global(self.X_train_global, self.Gamma_global, self.Beta_global,
                                                           self.M_global, self.Sigma_global)
        'prediction before update'

        s, h = self.decision_value(0, X_minibatch_global_BN, 1, self.N_batch)

        for n in range(self.N_batch):
            temp = (self.C1 - self.C0) * s[n, 0] * self.label_train[n, 0] - self.C1 * self.label_train[n, 0] + self.C0 * s[n, 0]
            Rn = np.zeros((Ch, 1))
            for p in range(N_te):
                Rn = Rn + np.dot(X_minibatch_global_BN[:, :, n], self.Q_global[:, p]).reshape((60, 1))
            Un = np.zeros(Te)
            for m in range(N_sp):
                Un = Un + np.dot(self.W_global[m, :], X_minibatch_global_BN[:, :, n])

            X_n_BN0 = (X_minibatch[:, :, n] - self.M_global) / np.sqrt(self.Sigma_global)

            if not np.isnan(temp) and abs(temp) < float('inf'):
                for m in range(N_sp):
                    for c in range(Ch):
                        delta_W_global[m, c] = delta_W_global[m, c] + 1 / self.N_batch * temp * Rn[c] / N_sp / N_te
                    for p in range(N_te):
                        delta_Gamma_global = delta_Gamma_global + 1 / self.N_batch * temp * self.W_global[m, :] @ X_n_BN0 @ self.Q_global[:, p].reshape((-1, 1))/N_sp/N_te
                        delta_Beta_global = delta_Beta_global + 1 / self.N_batch * temp * self.W_global[m, :] @  np.ones((Ch, Te)) @ self.Q_global[:, p].reshape((-1, 1))/N_sp/N_te
                for p in range(N_te):
                    for t in range(Te):
                        delta_Q_global[t, p] = delta_Q_global[t, p] + 1/self.N_batch * temp * Un[t]/N_sp/N_te
                delta_b_global = delta_b_global + temp / self.N_batch

            if self.crossentropy_flag:
                Crossentropy = Crossentropy - self.label_train[n, 0] * np.log(s[n, 0]) / self.N_batch - (1 - self.label_train[n, 0]) * np.log(1 - s[n, 0]) / self.N_batch

        self.W_global, self.mW_global, self.vW_global, lrw = self.ADAM(self.W_global, delta_W_global, self.mW_global, self.vW_global)
        self.Q_global, self.mQ_global, self.vQ_global, lrq = self.ADAM(self.Q_global, delta_Q_global, self.mQ_global, self.vQ_global)
        self.b_global, self.mb_global, self.vb_global, lrb = self.ADAM(self.b_global, delta_b_global, self.mb_global, self.vb_global)
        self.Beta_global, self.mBeta_global, self.vBeta_global, lrbeta = self.ADAM(self.Beta_global, delta_Beta_global, self.mBeta_global, self.vBeta_global)
        self.Gamma_global, self.mGamma_global, self.vGamma_global, lrgamma = self.ADAM(self.Gamma_global, delta_Gamma_global, self.mGamma_global, self.vGamma_global)
        lr = lrw.copy()
        mStepSize = np.mean(lr)
        stdStepSize = np.std(lr)

        'Validation'
        # print('Subject %d Model %d/300 idx_interation %d' % (idx_sub, idx_model, idx_batch))
        if self.validation_flag and\
                ((idx_model % self.validation_step == 0 and idx_batch == self.N_iteration - 1) or (idx_model == 1 and idx_batch == self.N_iteration - 1)):
            Accvalidation, tpr, fpr, auc = self.validation(idx_model)

        else:
            Accvalidation = 0
            tpr = 0
            fpr = 0
            auc = 0

        return (
            Crossentropy,
            Accvalidation,
            tpr,
            fpr,
            auc,
            mStepSize, stdStepSize
        )

    def generate_submodel(self, G_k, H_k, idx_batch, idx_model):
        it = idx_batch
        idx_conv = idx_model - 1
        Nv = np.shape(self.X_validation)[0]
        N_win = idx_conv
        'Batch normalization // 1. mean and sigma for whold batch ! # NOT mini batch !'
        idx_BN = idx_batch
        m_old = self.M_local[idx_conv - 1, :]
        sigma_old = self.Sigma[idx_conv - 1, :]
        m_now = np.mean(self.X_train[:, :, idx_conv - 1], axis=0) #batch mean
        self.M_local[idx_conv - 1, :] = self.slidemean(m_old, m_now, idx_BN * self.N_batch, self.N_batch)  # update mean
        # if idx_BN == 0:
        # 	M[idx_conv - 1, :] = MARK55_slidemean(m_old, m_now, (idx_BN ) * N, N)  # update mean
        # else:
        # 	M[idx_conv - 1, :] = MARK55_slidemean(m_old, m_now, (idx_BN - 1) * N, N) #update mean
        sigma_now = np.zeros(self.T_local)

        for n in range(self.N_batch):
            sigma_now = sigma_now + (self.X_train[n, :, idx_conv-1] - self.M_local[idx_conv-1, :])**2
        sigma_now = sigma_now / self.N_batch # batch sigma
        self.Sigma[idx_conv - 1, :] = self.slidemean(sigma_old, sigma_now, idx_BN * self.N_batch, self.N_batch)  # update sigma

        sigma = self.Sigma[idx_conv-1, :]
        sigma[np.where(sigma == 0)] = np.mean(sigma)
        self.Sigma[idx_conv-1, :] = sigma.copy()
        'Iteration and update'
        Crossentropy = 0 # zeros(1, N_iteration);

        f_k = np.zeros((self.N_batch, 1))
        'Iteration'
        delta_w = 2 * self.eta * self.W_local[idx_conv-1, :]
        delta_w[0] = 0
        delta_gamma = 0
        delta_beta = 0
        X_minibatch_BN = np.zeros((self.N_batch, self.T_local, idx_conv))
        X_minibatch = self.X_train[:, :, idx_conv-1]

        X_minibatch_BN_global = self.batchnormalize_global(self.X_train_global, self.Gamma_global, self.Beta_global, self.M_global, self.Sigma_global)
        for k in range(idx_conv):
            X_minibatch_BN[:, :, k] = self.batchnormalize(self.X_train[:, :, k], self.Gamma[k], self.Beta[k], self.M_local[k, :], self.Sigma[k, :])
        for n in range(self.N_batch):
            f_k[n, 0] = self.W_local[idx_conv-1, 1: self.T_local+1] @ X_minibatch_BN[n, :, idx_conv-1].T + self.W_local[idx_conv-1, 0]
        # sum(W[idx_conv - 1, :] * np.concatenate([[1], X_minibatch_BN[n, :, idx_conv - 1]]))
        for n in range(self.N_batch):
            temp = G_k[n, 0] + H_k[n, 0] * f_k[n, 0]
            if not np.isnan(temp) and np.abs(temp) < float('inf'):
                for t in range(self.T_local):
                    delta_w[t+1] = delta_w[t+1] + 1/self.N_batch * temp * X_minibatch_BN[n, t, idx_conv-1]
                    delta_beta = delta_beta + 1/self.N_batch * temp * self.W_local[idx_conv - 1, t+1]
                    delta_gamma = delta_gamma + 1 / self.N_batch * temp * self.W_local[idx_conv - 1, t + 1] * (
                                X_minibatch[n, t] - self.M_local[idx_conv - 1, t]) / np.sqrt(self.Sigma[idx_conv - 1, t])

                delta_w[0] = delta_w[0] + temp/self.N_batch
            'Crossentropy'
            if self.crossentropy_flag:
                N_sp = np.shape(self.W_global)[0]
                N_te = np.shape(self.Q_global)[1]
                f_sigma = 0
                for m in range(N_sp):
                    for p in range(N_te):
                        f_sigma = f_sigma + self.W_global[m, :] @ X_minibatch_BN_global[:, :, n] @ self.Q_global[:, p]/N_sp/N_te
                f_sigma = f_sigma + self.b_global
                f_sigma = f_sigma * self.gstf_weight
                for k in range(idx_conv):
                    f = self.W_local[k, 1:self.T_local+1] @ X_minibatch_BN[n, :, k].T + self.W_local[k, 0]
                    f_sigma = f_sigma + self.lr_model[k] * f
                h_k = f_sigma.copy()
                s = 1/(1 + np.exp(-1 * h_k))
                Crossentropy = Crossentropy - self.label_train[n, 0] * np.log(s) / self.N_batch - (1 - self.label_train[n, 0]) * np.log(
                    1 - s) / self.N_batch

        self.W_local[idx_conv - 1, :], self.mW[idx_conv - 1, :], self.vW[idx_conv - 1, :], lrw = self.ADAM(self.W_local[idx_conv - 1, :], delta_w,
                                                                                             self.mW[idx_conv - 1, :],
                                                                                             self.vW[idx_conv - 1, :])

        self.Beta[idx_conv - 1], self.mBeta[idx_conv - 1], self.vBeta[idx_conv - 1], lrbeta = self.ADAM(self.Beta[idx_conv - 1],
                                                                                                delta_beta,
                                                                                                self.mBeta[idx_conv - 1],
                                                                                                self.vBeta[idx_conv - 1])

        self.Gamma[idx_conv - 1], self.mGamma[idx_conv - 1], self.vGamma[idx_conv - 1], lrgamma = self.ADAM(self.Gamma[idx_conv - 1],
                                                                                                    delta_gamma,
                                                                                                    self.mGamma[idx_conv - 1],
                                                                                                    self.vGamma[idx_conv - 1])
        lr = lrw.copy()  # 存疑  lr = lrw[:]
        mStepSize = np.mean(lr)
        stdStepSize = np.std(lr)

        # print('Crossentropy = %f' % Crossentropy)
        'Validation'
        # print('Subject %d Model %d/300 idx_interation %d' % (idx_sub, idx_model, idx_BN))
        if self.validation_flag and (idx_model % self.validation_step) == 0 and idx_batch == self.N_iteration - 1:
            Accvalidation, tpr, fpr, auc = self.validation(idx_model)

        else:
            Accvalidation = 0
            tpr = 0
            fpr = 0
            auc = 0
        return (
            Crossentropy,
            Accvalidation,
            tpr,
            fpr,
            auc,
            mStepSize, stdStepSize
        )

    def load_model(self):
        filename = 'Model_' + str(self.sub_idx) + '.npz'
        model = np.load(os.path.join(self.model_path, filename))
        self.W_local = model['W_local']
        self.Gamma = model['Gamma']
        self.Beta = model['Beta']
        self.Sigma = model['Sigma']
        self.M_local = model['M_local']
        self.lr_model = model['lr_model']
        self.W_global = model['W_global']
        self.Q_global = model['Q_global']
        self.b_global = model['b_global']
        self.Gamma_global = model['Gamma_global']
        self.Beta_global = model['Beta_global']
        self.Sigma_global = model['Sigma_global']
        self.M_global = model['M_global']
        self.I_sort = model['conv_sort']
        self.acc = model['Accvalidation_all']
        self.tpr = model['tpr_all']
        self.fpr = model['fpr_all']
        self.auc = model['auc_all']
        self.N_model = np.shape(self.W_local)[0]

    def test(self, testset):
        self.load_model()
        self.get_3Dconv()
        Tset_test, NTset_test, Tset_test_global, NTset_test_global, K1t, K2t = \
            self.get_data(testset)

        X_test = np.concatenate((Tset_test, NTset_test), axis=0)
        self.X_test_global = np.concatenate((Tset_test_global, NTset_test_global), axis=2)
        self.label_test = np.concatenate((np.ones((K1t, 1)), np.zeros((K2t, 1))), axis=0)
        self.X_test = X_test[:, :, self.I_sort[:self.N_model]]

    def train_model(self):
        self.get_3Dconv()
        self.Tset_train, self.NTset_train, self.Tset_train_global, self.NTset_train_global, self.K1, self.K2 = \
            self.get_data(self.trainset)

        self.get_order_step()
        self.NTset_train = self.NTset_train[:, :, self.I_sort[:self.N_model]]
        self.Tset_train = self.Tset_train[:, :, self.I_sort[:self.N_model]]

        'Downsampling negative samples'
        idx_nontarget_train = [i for i in range(self.K2)]
        random.shuffle(idx_nontarget_train)
        idx_selected_2 = idx_nontarget_train[:self.K1]
        self.NTset_train = self.NTset_train[idx_selected_2, :, :]
        self.NTset_train_global = self.NTset_train_global[:, :, idx_selected_2]
        self.K2 = np.shape(idx_selected_2)[0]

        Tset_validation, NTset_validation, Tset_validation_global, NTset_validation_global, self.K1v, self.K2v = \
            self.get_data(self.validationset)

        self.X_validation = np.concatenate((Tset_validation, NTset_validation), axis=0)
        self.X_validation = self.X_validation[:, :, self.I_sort[:self.N_model]]

        self.X_validation_global = np.concatenate((Tset_validation_global, NTset_validation_global), axis=2)

        self.Nb = np.min([self.Nb, self.K1])
        self.label_validation = np.concatenate((np.ones((self.K1v, 1)), np.zeros((self.K2v, 1))), 0)

        self.N_batch = self.Nb * 2
        self.label_train = np.concatenate((np.ones((self.Nb, 1)), np.zeros((self.Nb, 1))), 0)
        self.label_all = np.concatenate((np.ones((self.K1, 1)), np.zeros((self.K2, 1))), 0)

        '''INITIALIZATION'''
        N_sp = 1
        N_te = 1

        self.W_global = 0.01 + 0.01 * np.random.rand(N_sp, np.shape(self.Tset_train_global)[0])
        self.Q_global = 0.01 + 0.01 * np.random.rand(np.shape(self.Tset_train_global)[1], N_te)
        self.b_global = 0
        self.Gamma_global = 1
        self.Beta_global = 0
        self. Sigma_global = np.zeros((np.shape(self.Tset_train_global)[0], np.shape(self.Tset_train_global)[1]))
        self.M_global = np.zeros((np.shape(self.Tset_train_global)[0], np.shape(self.Tset_train_global)[1]))

        self.mW_global = self.W_global * 0
        self.vW_global = self.mW_global.copy()

        self.mQ_global = self.Q_global * 0
        self.vQ_global = self.mQ_global.copy()

        self.mb_global = 0
        self.vb_global = 0

        self.mBeta_global = 0
        self.vBeta_global = 0

        self.mGamma_global = 0
        self.vGamma_global = 0

        'local model'

        self.W_local = np.zeros((self.N_model, self.T_local + 1))  # 0.01 + 0.01 * np.random.rand(N_model, T_local + 1)
        self.M_local = np.zeros((self.N_model, self.T_local))  # continue from the same window
        self.Sigma = np.zeros((self.N_model, self.T_local))  # continue from the same window
        self.Gamma = np.ones((self.N_model, 1))
        self.Beta = np.zeros((self.N_model, 1))

        self.mW = self.W_local * 0
        self.vW = self.mW.copy()
        self.mBeta = self.Beta.copy()
        self.vBeta = self.Beta.copy()
        self.mGamma = self.Beta.copy()
        self.vGamma = self.Beta.copy()

        '''
        XGB-DIM
        '''
        idx_model = 0
        for idx_conv in range(self.N_model + 1):
            if idx_conv == 0:
                idx_model += 1  #
                for idx_iteration in range(self.N_iteration):
                    idx_sample_target_train = [i for i in range(self.K1)]
                    random.shuffle(idx_sample_target_train)
                    select_1_idx = idx_sample_target_train[:self.Nb]

                    idx_sample_nontarget_train = [i for i in range(self.K2)]
                    random.shuffle(idx_sample_nontarget_train)
                    select_2_idx = idx_sample_nontarget_train[:self.Nb]

                    self.X_train_global = np.concatenate(
                        (self.Tset_train_global[:, :, select_1_idx], self.NTset_train_global[:, :, select_2_idx]), 2)

                    Crossentropy, Accvalidation, tpr, fpr, auc, mStepSize, stdStepSize =\
                        self.MBGD_global_STF(idx_iteration, idx_model)

                    Crossentropy_all = Crossentropy
                    Accvalidation_all = Accvalidation
                    tpr_all = tpr
                    fpr_all = fpr
                    auc_all = auc

                print('Subject %d Model %d/300 done !' % (self.sub_idx, idx_model))
                print('ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))

            else:  #
                G_k_T, H_k_T = self.get_GH(self.Tset_train_global, self.Tset_train[:, :, :idx_conv],
                                                np.ones((np.shape(self.Tset_train)[0], 1)), idx_model)

                G_k_N, H_k_N = self.get_GH(self.NTset_train_global, self.NTset_train[:, :, :idx_conv],
                                                np.zeros((np.shape(self.NTset_train)[0], 1)), idx_model)

                idx_model = idx_model + 1

                Crossentropy = np.zeros(self.N_iteration)
                Accvalidation = Crossentropy.copy()
                tpr = Crossentropy.copy()
                fpr = Crossentropy.copy()
                auc = Crossentropy.copy()

                for idx_iteration in range(self.N_iteration):
                    idx_sample_target_train = [i for i in range(self.K1)]
                    random.shuffle(idx_sample_target_train)
                    select_1_idx = idx_sample_target_train[:self.Nb]

                    idx_sample_nontarget_train = [i for i in range(self.K2)]
                    random.shuffle(idx_sample_nontarget_train)
                    select_2_idx = idx_sample_nontarget_train[:self.Nb]

                    self.X_train_global = np.concatenate(
                        [self.Tset_train_global[:, :, select_1_idx], self.NTset_train_global[:, :, select_2_idx]], axis=2)
                    self.X_train = np.concatenate(
                        (self.Tset_train[select_1_idx, :, :idx_conv], self.NTset_train[select_2_idx, :, :idx_conv]), axis=0)
                    G_k = np.concatenate((G_k_T[select_1_idx], G_k_N[select_2_idx]), axis=0)
                    H_k = np.concatenate((H_k_T[select_1_idx], H_k_N[select_2_idx]), axis=0)

                    [Crossentropy[idx_iteration], Accvalidation[idx_iteration], tpr[idx_iteration], fpr[idx_iteration], auc[idx_iteration],
                    mStepSize, stdStepSize] = self.generate_submodel(G_k, H_k, idx_iteration, idx_model)

                print('Subject %d Model %d/300 done !' % (self.sub_idx, idx_model))

            if self.validation_flag and idx_model % self.validation_step == 0 and idx_model != 0:
                print('ACC %f TPR %f FPR %f AUC %f' % (
                Accvalidation[self.N_iteration - 1], tpr[self.N_iteration - 1], fpr[self.N_iteration - 1], auc[self.N_iteration - 1]))
                Crossentropy_all = np.append(Crossentropy_all,
                                             np.ones(self.validation_step) * self.CrossEntropy(np.concatenate((self.Tset_train_global, self.NTset_train_global), axis=2),
                                                  np.concatenate(
                                                      (self.Tset_train[:, :, :idx_conv], self.NTset_train[:, :, :idx_conv]),
                                                      axis=0), self.label_all, idx_model))

                Accvalidation_all = np.append(Accvalidation_all, np.ones((1, 10)) * Accvalidation[self.N_iteration - 1])
                tpr_all = np.append(tpr_all, np.ones((1, self.validation_step)) * tpr[self.N_iteration - 1])
                fpr_all = np.append(fpr_all, np.ones((1, self.validation_step)) * fpr[self.N_iteration - 1])
                auc_all = np.append(auc_all, np.ones((1, self.validation_step)) * auc[self.N_iteration - 1])

        print('Model Training Finished !')
        filename = 'Model_' + str(self.sub_idx) + '.npz'
        np.savez(os.path.join(self.model_path, filename), W_global=self.W_global, Q_global=self.Q_global, b_global=self.b_global,
                 Gamma_global=self.Gamma_global, Beta_global=self.Beta_global, Sigma_global=self.Sigma_global, M_global=self.M_global,
                 W_local=self.W_local, Gamma=self.Gamma, Beta=self.Beta, Sigma=self.Sigma, M_local=self.M_local, lr_model=self.lr_model,
                 conv_sort=self.I_sort, Accvalidation_all=Accvalidation_all, tpr_all=tpr_all, fpr_all=fpr_all,
                 auc_all=auc_all)
        return Crossentropy_all, Accvalidation_all, tpr_all, fpr_all, auc_all



'''
XGBDIM.py
Author: Bowen Li, Tsinghua University
'''
import numpy as np
import os
import h5py
import torch as tc
import torch.optim as optim
import torch.nn as nn
import time
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
random_downsampling_flag: whether to use the random downsampling for the negative samples
'''
class model_GSTF(nn.Module):
    def __init__(self, N_chan, N_time):
        super(model_GSTF, self).__init__()
        self.W_global = nn.Parameter(0.02 * tc.randn(N_chan, 1))
        self.Q_global = nn.Parameter(0.02 * tc.randn(N_time, 1))
        self.b_global = nn.Parameter(tc.zeros(1).float())
        self.Gamma_global = nn.Parameter(tc.ones(1).float())
        self.Beta_global = nn.Parameter(tc.zeros(1).float())
        self.h = 0

    def forward(self, x):
        # input x: (60, 250, batch_size) -> (batch_size, 250, 60)
        x = self.Gamma_global * x + self.Beta_global
        h = tc.bmm(x.permute(2, 1, 0), self.W_global.repeat(x.shape[2], 1, 1))
        self.h = tc.mm(h.squeeze(-1), self.Q_global) + self.b_global
        s = tc.sigmoid(self.h)
        return s

class model_local(nn.Module):
    def __init__(self, T_local):
        super(model_local, self).__init__()
        self.w_local = nn.Parameter(0.02 * tc.randn(T_local, 1))
        self.b_local = nn.Parameter(tc.zeros(1).float())
        self.gamma_local = nn.Parameter(tc.ones(1).float())
        self.beta_local = nn.Parameter(tc.zeros(1).float())
        self.fk = 0

    def forward(self, x):
        # input x: (batch_size, T_local)
        x = self.gamma_local * x + self.beta_local
        f = tc.mm(x, self.w_local) + self.b_local
        self.fk = f
        return f

class loss_model_local(nn.Module):
    def __init__(self, G_k, H_k):
        super(loss_model_local, self).__init__()
        self.G_k = G_k
        self.H_k = H_k

    def forward(self, x):
        # input x: (batch_size, 1)
        # G_k: (batch_size, 1)
        # H_k: (batch_size, 1)
        return tc.mean(self.G_k * x + 0.5 * self.H_k * (x ** 2))

class XGBDIM():

    def __init__(self, data_path, sub_idx, trainset, validationset,
                 model_path,
                 n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
                 eta, alpha, Nb, N_iteration, C1, C0, max_N_model, gstf_weight,
                 validation_flag, validation_step, crossentropy_flag, random_downsampling_flag):

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
        self.random_downsampling_flag = random_downsampling_flag
        temp = np.array(range(1, 55))
        self.channel_loc = np.array([temp[0:9], temp[9:18], temp[18:27], temp[27:36], temp[36:45], temp[45:54]])

        self.n_cutpoint = n_cutpoint # number of points before stimulus
        self.beta1 = 0.9
        self.beta2 = 0.999

    def get_3Dconv(self):
        Nx_channel = np.shape(self.channel_loc)[1]
        channel_xst = np.arange(0, Nx_channel - self.chan_xlen + 1, self.step_x)

        Ny_channel = np.shape(self.channel_loc)[0]
        channel_yst = np.arange(0, Ny_channel - self.chan_ylen + 1, self.step_y)

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
        # self.mask = np.zeros((60, 250, self.N_model))
        # idx_conv = 0
        # for idx_chan in range(self.N_chanwin):
        #     for idx_win in range(self.N_win):
        #         self.mask[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
        #                     self.window_st[idx_win] - 1:self.window_ov[idx_win], idx_conv] = 1
        #         idx_conv += 1
        # # self.mask.dtype = 'bool'
        # self.mask = tc.from_numpy(self.mask).bool().cuda(0)
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
                X1 = np.float32(np.transpose(data['X1']))
                X2 = np.float32(np.transpose(data['X2']))
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

    def get_3D_cuboids(self, X1, X2, K1, K2): # torch.masked_select(input,mask) is FASTER?
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
                        k])
                    Tset[k, :, idx_conv] = cup.T.flatten()
                for k in range(K2):
                    cup = np.squeeze(
                        X2[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
                        self.window_st[idx_win] - 1:self.window_ov[idx_win],
                        k])
                    NTset[k, :, idx_conv] = cup.T.flatten()
        return Tset, NTset

    def preprocess(self, X1, X2, K1, K2):
        X1 = np.delete(X1, list(range(self.n_cutpoint)), axis=1)
        X2 = np.delete(X2, list(range(self.n_cutpoint)), axis=1)
        for k in range(K1):
            X1[:, :, k] = X1[:, :, k] - np.mean(X1[:, :, k], axis=1)[:, np.newaxis]
        for k in range(K2):
            X2[:, :, k] = X2[:, :, k] - np.mean(X2[:, :, k], axis=1)[:, np.newaxis]
        print('EEG processed !')
        return X1, X2

    def get_data(self, dataset):
        start_time = time.time()
        X1, X2, K1, K2 = self.read_data(dataset)
        # X1 = tc.tensor(X1)
        # X2 = tc.tensor(X2)
        X1, X2 = self.preprocess(X1, X2, K1, K2)
        print('Time cost: ', time.time() - start_time)
        start_time = time.time()
        Tset, NTset = self.get_3D_cuboids(X1, X2, K1, K2)

        Tset_global = X1.copy()
        NTset_global = X2.copy()
        print('EEG reshaped !')
        print('Time cost: ', time.time() - start_time)
        return Tset, NTset, Tset_global, NTset_global, K1, K2

    def get_order_step(self):
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
                    np.linalg.pinv(R1 + R2) @ np.matrix(M2[idx_conv, :] - M1[idx_conv, :]).T @ np.matrix(
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

    def batchnormalize(self, X_local, m, sigma):
        Ns = X_local.shape[0]
        X_local_BN = (X_local - m.repeat(Ns, 1)) / tc.sqrt(sigma.repeat(Ns, 1))
        return X_local_BN

    def batchnormalize_global(self, X_global, M_global, Sigma_global):
        # [Ch, Te, Ns] = X_global.shape
        if X_global.dim == 2:
            X_global_BN = (X_global - M_global) / tc.sqrt(Sigma_global)
        else:
            X_global_BN = (X_global - M_global.unsqueeze(2)) / tc.sqrt(Sigma_global.unsqueeze(2))

        return X_global_BN

    def slidemean(self, x_old, x_now, n_old, n_now):
        x_update = (x_old * n_old + x_now * n_now) / (n_old + n_now)
        return x_update


    def decision_value(self, X_minibatch_BN, X_minibatch_BN_global, model_all, Ns):
        # h = tc.zeros((Ns, 1)).cuda(0)
        N_model = len(model_all)
        gstf = model_all[0]
        gstf(X_minibatch_BN_global)
        h = gstf.h
        if N_model > 1:
            h = self.gstf_weight * h

        for idx_model in range(1, N_model):
            local_model = model_all[idx_model]
            local_model(X_minibatch_BN[:, :, idx_model - 1])
            f = local_model.fk
            h = h + self.lr_model[idx_model - 1, 0] * f

        s = 1 / (1 + tc.exp(-1 * h))
        return s, h

    def validation(self, model_all):
        N_model = len(model_all)
        Ns = self.X_validation_global.shape[2]
        X_global_BN = self.batchnormalize_global(self.X_validation_global, self.M_global, self.Sigma_global)
        X_minibatch_BN = tc.zeros((Ns, self.T_local, N_model - 1)).float().cuda(0)
        for idx_conv in range(N_model - 1):
            X_minibatch_BN[:, :, idx_conv] = self.batchnormalize(self.X_validation[:, :, idx_conv],
                                                                 self.M_local[idx_conv, :], self.Sigma[idx_conv, :])

        s, h = self.decision_value(X_minibatch_BN, X_global_BN, model_all, Ns)

        idx_1 = tc.where(tc.eq(self.label_validation, 1))[0]
        idx_2 = tc.where(tc.eq(self.label_validation, 0))[0]

        y_predicted_final = s.clone()
        y_predicted_final[s >= 0.5] = 1
        y_predicted_final[s < 0.5] = 0

        acc = (y_predicted_final == self.label_validation).sum() / y_predicted_final.shape[0]

        n_positive = self.label_validation.eq(1).sum()
        n_negative = self.label_validation.eq(0).sum()
        n_tp = (y_predicted_final.T * self.label_validation.T).sum()
        n_fp = ((y_predicted_final.T == 1) * (self.label_validation.T == 0)).sum()

        tpr = n_tp / n_positive
        fpr = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(self.label_validation.cpu().numpy(), s.detach().cpu().numpy())
        auc = metrics_auc(fpr_1, tpr_1)

        return acc, tpr, fpr, auc

    def get_GH(self, X_global, X_local, label, model_all):
        N_model = len(model_all)
        Ch, Te, Ns = X_global.shape
        X_global_BN_1 = self.batchnormalize_global(X_global, self.M_global, self.Sigma_global)
        # X_minibatch_BN = np.zeros((Ns, self.T_local, N_model - 1))
        if N_model == 1:
            s, self.h_GH_temp = self.decision_value(0, X_global_BN_1, model_all, Ns)
            s = 1 / (1 + tc.exp(-self.h_GH_temp))
            self.h_GH_temp = self.h_GH_temp * self.gstf_weight
        else:
            X_minibatch_BN = self.batchnormalize(X_local[:, :, N_model-2], self.M_local[N_model-2, :], self.Sigma[N_model-2, :])
            local_model = model_all[-1]
            local_model(X_minibatch_BN)
            f_k = local_model.fk
            self.h_GH_temp = self.h_GH_temp + self.lr_model[N_model-2, 0] * f_k

            s = 1 / (1 + tc.exp(-self.h_GH_temp))
        G_k = (self.C1 - self.C0) * s * label - self.C1 * label + self.C0 * s
        H_k = ((self.C1 - self.C0) * label + self.C0) * s * (1 - s)

        # self.h_T_temp = self.h_GH_temp[label == 1].view(tc.sum(label == 1), 1)
        # self.h_N_temp = self.h_GH_temp[label == 0].view(tc.sum(label == 0), 1)
        G_k_T = G_k[label == 1].view(tc.sum(label == 1), 1)
        H_k_T = H_k[label == 1].view(tc.sum(label == 1), 1)
        G_k_N = G_k[label == 0].view(tc.sum(label == 0), 1)
        H_k_N = H_k[label == 0].view(tc.sum(label == 0), 1)
        return G_k_T, H_k_T, G_k_N, H_k_N

    def update_BN_global(self, X_train_global_BN, idx_batch):

        m_old = self.M_global.clone()
        sigma_old = self.Sigma_global.clone()

        m_now = tc.mean(X_train_global_BN, dim=2)  # batch mean  #check
        idx_BN = idx_batch
        self.M_global = self.slidemean(m_old, m_now, idx_BN * self.N_batch, self.N_batch)
        sigma_now = tc.mean((X_train_global_BN - self.M_global.unsqueeze(2)) ** 2, dim=2)
        self.Sigma_global = self.slidemean(sigma_old, sigma_now, idx_BN * self.N_batch, self.N_batch)  # update sigma

    def update_BN_local(self, X_train_local_BN, idx_batch, N_model):
        idx_BN = idx_batch
        m_old = self.M_local[N_model - 2, :].clone()
        sigma_old = self.Sigma[N_model - 2, :].clone()
        m_now = tc.mean(X_train_local_BN, dim = 0)  # batch mean
        self.M_local[N_model - 2, :] = self.slidemean(m_old, m_now, idx_BN * self.N_batch, self.N_batch)  # update mean
        m = self.M_local[N_model - 2, :].clone().detach()

        sigma_now = tc.mean((X_train_local_BN - self.M_local[N_model - 2, :]) ** 2, axis=0)
         # update sigma
        sigma = self.slidemean(sigma_old, sigma_now, idx_BN * self.N_batch,
                                                     self.N_batch)
        sigma[sigma == 0] = tc.mean(sigma)
        self.Sigma[N_model - 2, :] = sigma.clone()


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
        print('Model of Subject %d is loaded' % self.sub_idx)
        self.get_3Dconv()
        Tset_test, NTset_test, Tset_test_global, NTset_test_global, K1t, K2t = \
            self.get_data(testset)

        X_test = np.concatenate((Tset_test, NTset_test), axis=0)
        X_test_global = np.concatenate((Tset_test_global, NTset_test_global), axis=2)
        label_test = np.concatenate((np.ones((K1t, 1)), np.zeros((K2t, 1))), axis=0)
        X_test = X_test[:, :, self.I_sort[:self.N_model]]
        print('Samples obtained!')
        Ns = np.shape(X_test_global)[2]

        X_global_BN = self.batchnormalize_global(X_test_global, self.Gamma_global, self.Beta_global,
                                                 self.M_global, self.Sigma_global)
        X_minibatch_BN = np.zeros((Ns, self.T_local, self.N_model - 1))
        for idx_conv in range(self.N_model - 1):
            X_minibatch_BN[:, :, idx_conv] = self.batchnormalize(X_test[:, :, idx_conv],
                                                                 self.Gamma[idx_conv],
                                                                 self.Beta[idx_conv],
                                                                 self.M_local[idx_conv, :], self.Sigma[idx_conv, :])

        s, h = self.decision_value(X_minibatch_BN, X_global_BN, self.N_model, Ns)

        idx_1 = np.where(np.array(label_test) == 1)
        idx_2 = np.where(np.array(label_test) == 0)

        y_predicted_final = s.copy()
        y_predicted_final[np.where(y_predicted_final >= 0.5)] = int(1)
        y_predicted_final[np.where(y_predicted_final < 0.5)] = int(0)

        acc = np.sum((y_predicted_final == label_test) != 0) / np.shape(y_predicted_final)[0]

        n_positive = np.sum(label_test == 1)
        n_negative = np.sum(label_test == 0)
        n_tp = np.sum(y_predicted_final.T * label_test.T)
        n_fp = np.sum((y_predicted_final.T == 1) * (label_test.T == 0))

        tpr = n_tp / n_positive
        fpr = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(label_test, s)
        auc = metrics_auc(fpr_1, tpr_1)
        ba = (tpr + (1 - fpr)) / 2
        return ba, acc, tpr, fpr, auc

    def train_model(self):
        self.get_3Dconv()
        self.Tset_train, self.NTset_train, self.Tset_train_global, self.NTset_train_global, self.K1, self.K2 = \
            self.get_data(self.trainset)

        self.get_order_step()
        self.NTset_train = self.NTset_train[:, :, self.I_sort[:self.N_model]]
        self.Tset_train = self.Tset_train[:, :, self.I_sort[:self.N_model]]

        self.Tset_train = tc.from_numpy(self.Tset_train).float().cuda(0)
        self.NTset_train = tc.from_numpy(self.NTset_train).float().cuda(0)
        self.Tset_train_global = tc.from_numpy(self.Tset_train_global).float().cuda(0)
        self.NTset_train_global = tc.from_numpy(self.NTset_train_global).float().cuda(0)
        self.lr_model = tc.from_numpy(self.lr_model).view(-1,1).float().cuda(0)

        'Downsampling negative samples'
        if self.random_downsampling_flag:
            idx_nontarget_train = [i for i in range(self.K2)]
            random.shuffle(idx_nontarget_train)
            idx_selected_2 = idx_nontarget_train[:self.K1]
        else:
            idx_selected_2 = tc.arange(0, self.K2, round(self.K2 / self.K1))

        self.NTset_train = self.NTset_train[idx_selected_2, :, :]
        self.NTset_train_global = self.NTset_train_global[:, :, idx_selected_2]
        self.K2 = np.shape(idx_selected_2)[0]

        Tset_validation, NTset_validation, Tset_validation_global, NTset_validation_global, self.K1v, self.K2v = \
            self.get_data(self.validationset)

        Tset_validation = tc.from_numpy(Tset_validation).float().cuda(0)
        NTset_validation = tc.from_numpy(NTset_validation).float().cuda(0)
        Tset_validation_global = tc.from_numpy(Tset_validation_global).float().cuda(0)
        NTset_validation_global = tc.from_numpy(NTset_validation_global).float().cuda(0)
        self.I_sort = tc.from_numpy(self.I_sort.copy()).cuda(0)
        self.X_validation = tc.cat([Tset_validation, NTset_validation], dim=0)
        self.X_validation = self.X_validation[:, :, self.I_sort[:self.N_model]]

        self.X_validation_global = tc.cat([Tset_validation_global, NTset_validation_global], dim=2)

        self.Nb = np.min([self.Nb, self.K1])
        self.label_validation = tc.cat((tc.ones((self.K1v, 1)), tc.zeros((self.K2v, 1))), dim=0).float().cuda(0)

        self.N_batch = self.Nb * 2
        self.label_train = tc.cat([tc.ones((self.Nb, 1)), tc.zeros((self.Nb, 1))], dim=0).float().cuda(0)
        self.label_all = tc.cat((tc.ones((self.K1, 1)), tc.zeros((self.K2, 1))), dim=0).float().cuda(0)

        X_train_global_all = tc.cat((self.Tset_train_global, self.NTset_train_global), dim=2)
        X_train_local_all = tc.cat((self.Tset_train, self.NTset_train),dim=0)
        '''INITIALIZATION'''
        N_sp = 1
        N_te = 1

        self.Sigma_global = tc.zeros((self.Tset_train_global.shape[0], self.Tset_train_global.shape[1])).cuda(0)
        self.M_global = tc.zeros((self.Tset_train_global.shape[0], self.Tset_train_global.shape[1])).cuda(0)

        'local model'
        self.M_local = tc.zeros((self.N_model, self.T_local)).float().cuda(0)  # continue from the same window
        self.Sigma = tc.zeros((self.N_model, self.T_local)).float().cuda(0)  # continue from the same window

        '''
        XGB-DIM
        '''
        model_all = []
        idx_model = 0
        crossentropy = nn.CrossEntropyLoss()
        for idx_conv in range(self.N_model + 1):
            if idx_conv == 0:
                start_time = time.time()
                model= model_GSTF(self.Tset_train_global.shape[0], self.Tset_train_global.shape[1]).cuda(0)
                criterion = nn.CrossEntropyLoss()
                optimizer_global = optim.Adam(model.parameters(), lr=self.alpha)
                idx_model += 1  #
                for idx_iteration in range(self.N_iteration):
                    idx_sample_target_train = [i for i in range(self.K1)]
                    random.shuffle(idx_sample_target_train)
                    select_1_idx = idx_sample_target_train[:self.Nb]

                    idx_sample_nontarget_train = [i for i in range(self.K2)]
                    random.shuffle(idx_sample_nontarget_train)
                    select_2_idx = idx_sample_nontarget_train[:self.Nb]

                    X_train_global = tc.cat(
                        (self.Tset_train_global[:, :, select_1_idx], self.NTset_train_global[:, :, select_2_idx]), dim=2)
                    self.update_BN_global(X_train_global_all, idx_iteration)
                    X_train_global_BN = self.batchnormalize_global(X_train_global, self.M_global, self.Sigma_global)
                    optimizer_global.zero_grad()
                    outputs = model(X_train_global_BN) #X_train_global_BN, idx_batch
                    loss = criterion(outputs.view(1, -1), self.label_train.view(1, -1)) + self.eta * model.W_global.norm(2)\
                                                                                        + self.eta * model.Q_global.norm(2)
                    loss.backward()
                    optimizer_global.step()
                    print('Iteration: ', idx_iteration+1, '-- GSTF loss: ', loss.item())

                print('GSTF time cost: ', time.time() - start_time)
                print('Subject %d Model %d/300 done !' % (self.sub_idx, idx_model))
                model_all.append(model)
                # validation
                if self.validation_flag:
                    Accvalidation, tpr, fpr, auc = self.validation(model_all)
                    print('--------------ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))

            else:  #
                start_time = time.time()
                [G_k_T, H_k_T,
                 G_k_N, H_k_N] = self.get_GH(X_train_global_all, X_train_local_all, self.label_all, model_all)
                if self.crossentropy_flag:
                    s_temp = tc.sigmoid(self.h_GH_temp)
                    cross_entropy_temp = crossentropy(s_temp.view(1,-1), self.label_all.view(1,-1))
                    print('Cross_entropy: ', cross_entropy_temp.item())
                idx_model = idx_model + 1
                model = model_local(self.T_local).cuda(0)
                optimizer_local = optim.Adam(model.parameters(), lr=self.alpha)

                for idx_iteration in range(self.N_iteration):
                    idx_sample_target_train = [i for i in range(self.K1)]
                    random.shuffle(idx_sample_target_train)
                    select_1_idx = idx_sample_target_train[:self.Nb]

                    idx_sample_nontarget_train = [i for i in range(self.K2)]
                    random.shuffle(idx_sample_nontarget_train)
                    select_2_idx = idx_sample_nontarget_train[:self.Nb]
                    # self.X_train_global = tc.cat(
                    #     (self.Tset_train_global[:, :, select_1_idx], self.NTset_train_global[:, :, select_2_idx]), dim=2)
                    X_train_local = tc.cat(
                        (self.Tset_train[select_1_idx, :, idx_conv-1], self.NTset_train[select_2_idx, :, idx_conv-1]), dim=0)

                    self.update_BN_local(X_train_local, idx_iteration, idx_model)
                    X_train_local_BN = self.batchnormalize(X_train_local, self.M_local[idx_model-2, :], self.Sigma[idx_model-2, :])

                    G_k = tc.cat((G_k_T[select_1_idx], G_k_N[select_2_idx]), dim=0).clone().detach()
                    H_k = tc.cat((H_k_T[select_1_idx], H_k_N[select_2_idx]), dim=0).clone().detach()

                    criterion = loss_model_local(G_k, H_k)
                    optimizer_local.zero_grad()
                    outputs_local = model(X_train_local_BN)
                    loss_local = criterion(outputs_local) + self.eta * model.w_local.norm(2)
                    loss_local.backward()
                    optimizer_local.step()

                print('Subject %d Model %d/300 done ! Time cost %f' % (self.sub_idx, idx_model, time.time() - start_time))
                model_all.append(model)
            if self.validation_flag and idx_model % self.validation_step == 0 and idx_model != 0:
                Accvalidation, tpr, fpr, auc = self.validation(model_all)
                print('--------------ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))
                # print('ACC %f TPR %f FPR %f AUC %f' % (
                # Accvalidation[self.N_iteration - 1], tpr[self.N_iteration - 1], fpr[self.N_iteration - 1], auc[self.N_iteration - 1]))


        print('Model Training Finished !')
        # filename = 'Model_' + str(self.sub_idx) + '.npz'
        # np.savez(os.path.join(self.model_path, filename), W_global=self.W_global, Q_global=self.Q_global, b_global=self.b_global,
        #          Gamma_global=self.Gamma_global, Beta_global=self.Beta_global, Sigma_global=self.Sigma_global, M_global=self.M_global,
        #          W_local=self.W_local, Gamma=self.Gamma, Beta=self.Beta, Sigma=self.Sigma, M_local=self.M_local, lr_model=self.lr_model,
        #          conv_sort=self.I_sort, Accvalidation_all=Accvalidation_all, tpr_all=tpr_all, fpr_all=fpr_all,
        #          auc_all=auc_all)
        # return Crossentropy_all, Accvalidation_all, tpr_all, fpr_all, auc_all



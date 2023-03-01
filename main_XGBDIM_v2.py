from XGBDIM import XGBDIM
import numpy as np
'''
(data_path, sub_idx, trainset, validationset,
model_path,
n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
eta, alpha, Nb, N_iteration, C1, C0, max_N_sub_model, gstf_weight, 
validation_flag, validation_step, crossentropy_flag, random_downsampling)
'''
model_path = 'D:/XGBDIM/Model64'
xgb = XGBDIM(r'F:\XGB_for_V5_6', 1, np.array([1]), np.array([2, 3, 4]),
                 model_path,
                 50, 6, 3, 3, 3, 3,
                 0.5, 0.05, 100, 20, 1, 1, 299, 0.3, True, 30, False, False)

xgb.train_model()
# ba, acc, tpr, fpr, auc = xgb.test(np.array([2, 3, 4]))
# print(ba, acc, tpr, fpr, auc)

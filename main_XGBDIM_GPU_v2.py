from XGBDIM_GPU_v2 import XGBDIM
import numpy as np
'''
Next version: add a adaptive model weighting method
(data_path, sub_idx, trainset, validationset,
model_path,
n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
eta_global, eta_local, alpha_global, alpha_local, Nb, N_epoch, C1, C0, max_N_sub_model, gstf_weight, 
validation_flag, validation_step, crossentropy_flag, random_downsampling)
'''
model_path = 'D:/XGBDIM/Model64'
xgb = XGBDIM(r'F:\XGB_for_V5_6', 64, np.array([1]), np.array([2, 3, 4]),
                 model_path,
                 50, 6, 3, 3, 3, 3,
                 0.5, 0.5, 0.01, 0.05, 100, 20, 1, 1, 299, 0.3, True, 3, True, False)

xgb.train_model()


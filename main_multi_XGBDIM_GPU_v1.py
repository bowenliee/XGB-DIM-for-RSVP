from multi_XGBDIM_GPU_v1 import XGBDIM
import numpy as np
'''
(data_path, sub_idx, trainset, validationset,
model_path,
n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
eta_global, eta_local, alpha_global, alpha_local, Nb, N_epoch, C1, C0, max_N_sub_model, gstf_weight, 
validation_flag, validation_step, crossentropy_flag, random_downsampling)
'''
'''
Note:   the validation set in this study is used as the test set 
        for the purpose of comparing the performance of different number of sub-models.
        
        It is suggested that 
        the etas and alphas of global and local models are set in the range of [0.1, 0.5] and [0.01, 0.05] respectively.
        The choice will rarely affect the performance.
        
        The batch size is set to 100 in this study.
        The number of epochs is suggested to be set around 20. Too many epochs can cause the gradient to deplete prematurely.
        
        The number of groups is set to 20 in this study. You try more groups to see if it improves the performance, if your
        GPU is strong.
'''

model_path = 'D:/XGBDIM/Model64'
xgb = XGBDIM(r'F:\XGB_for_V5_6', 28, np.array([1]), np.array([2, 3, 4]),
                 model_path,
                 50, 6, 3, 3, 3, 3,
                 0.5, 0.1, 0.01, 0.05, 100, 20, 1, 1, 299, 0.3, True, 30, True, True, N_multiple = 20)

xgb.train_model()


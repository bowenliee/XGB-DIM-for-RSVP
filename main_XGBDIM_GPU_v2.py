from XGBDIM_GPU_v2 import XGBDIM
import numpy as np
'''
Next version: multi-XGB-DIM
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
'''

model_path = 'D:/XGBDIM/Model64'
xgb = XGBDIM(r'F:\XGB_for_V5_6', 64, np.array([1]), np.array([1]),
                 model_path,
                 50, 6, 3, 3, 3, 3,
                 0.5, 0.1, 0.01, 0.05, 100, 20, 1, 1, 299, 0.3, True, 30, True, False)

xgb.train_model()

ba, acc, tpr, fpr, auc = xgb.test(np.array([2, 3, 4]))
print('--------------BA %f ACC %f TPR %f FPR %f AUC %f' % (ba, acc, tpr, fpr, auc))


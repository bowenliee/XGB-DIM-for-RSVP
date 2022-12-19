#encoding:utf-8
def MARK56_XGBDIM_Sw(Tset_train,M1,NTset_train,M2,idx_conv):
	R1 = 0
	for k in range(K1):
		R1 = R1 + (Tset_train[k,:, idx_conv] - M1[idx_conv,:]) * (Tset_train[k,:, idx_conv]-M1[idx_conv,:])
	R1 = R1 / K1

	R2 = 0
	for k in range(K2):
		R2 = R2 + (NTset_train[k,:, idx_conv] - M2[idx_conv,:]) * (NTset_train[k,:, idx_conv]-M2[idx_conv,:])
	R2 = R2 / K2

	return R1,R2
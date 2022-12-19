#encoding:utf-8
import numpy as np

def MARK55_BN_global(X_global, Gamma_global, Beta_global, M_global, Sigma_global):
	[Ch, Te, Ns] = np.shape(X_global)
	X_global_BN = np.empty((Ch,Te,Ns)) * np.nan
	for n in range(Ns):

		if type(Gamma_global) == (int or float) and type(Beta_global) == (int or float):
			X_global_BN[:, :, n] = Gamma_global * (X_global[:, :, n] - M_global) / np.sqrt(Sigma_global) + Beta_global
		else:
			X_global_BN[:, :, n] = Gamma_global * (X_global[:, :, n] - M_global) / np.sqrt(Sigma_global) + Beta_global

	return X_global_BN
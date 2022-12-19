#encoding:utf-8
import numpy as np
def MARK55_BN(X_local,gamma,beta,m,sigma):
	Ns = np.shape(X_local)[0]
	X_local_BN = gamma * (X_local - np.ones((Ns, 1)) * m) / np.sqrt(np.ones((Ns, 1)) * sigma ) + beta
	return X_local_BN
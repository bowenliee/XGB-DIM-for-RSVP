import numpy as np

#encoding:utf-8
def MARK55_slidemean(x_old,x_now,n_old,n_now):
	x_update = (x_old * n_old + x_now * n_now) / (n_old + n_now)
	return x_update
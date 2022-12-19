import numpy as np
#encoding:utf-8
def MARK55_MBGD_ADAM(x,deltax,mx,vx,alpha,beta1,beta2,it):
	mx = beta1 * mx + (1 - beta1) * deltax
	vx = beta2 * vx + (1 - beta2) * (deltax * deltax)

	# 进行偏差校正
	beta1_t = 1 / 9
	beta2_t = 1 / 999
	m_ = mx  / (1 - beta1_t)
	v_ = vx  / (1 - beta2_t)

	solution = x - alpha / np.sqrt(v_ + alpha) * m_
	x = solution.copy()
	lrx = np.min(alpha / np.sqrt(v_ + alpha))

	return x, mx, vx, lrx








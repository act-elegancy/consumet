import numpy as np
import time
from win32com.client.gencache import EnsureDispatch
 
# Get the Excel Application COM object
xl = EnsureDispatch('Excel.Application')

def simulate(x):
	'''
	This function defines a function z = simulate(x), where x is a 2D variable and
	z is a 1D variable. In this example, simulate defines the Rosenbrock function
	and conducts the calculation in Excel to exemplify the Excel interface.
	'''
	
	tmp = 0
	for k in ['C3','C4']:
		xl.Range(k).Value = x[tmp]
		tmp += 1
	return np.array(xl.Range('E3').Value).flatten()

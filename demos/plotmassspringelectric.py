import numpy as np
data = np.loadtxt('/Users/edoardobonetti/Desktop/Uni/2025W/HPCDenseLinearAlgebra/ASC-EDO/build/output_test_ode_electric.txt', usecols=(0, 1, 2))
# print (data)

import matplotlib.pyplot as plt

plt.plot(data[:,0], data[:,1], label='capacitor voltage')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Electric Circuit Time Evolution')
plt.legend()
plt.grid()
plt.show()
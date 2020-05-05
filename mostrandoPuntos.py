import matplotlib.pyplot as plt
import numpy as np

y_corr = np.loadtxt('data/m_vista1.out')
x_corr = np.loadtxt('data/x_abajo1.out')
m_corr,b_corr = np.polyfit(x_corr,y_corr,1)
print(m_corr,b_corr)
plt.plot([x_corr], [y_corr], marker='o', markersize=3, color="red")
plt.grid()
plt.show()


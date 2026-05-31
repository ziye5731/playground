import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# 生成模拟混合信号
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 真实独立信号
s1 = np.sin(2 * time)            # 正弦
s2 = np.sign(np.sin(3 * time))   # 方波
S = np.c_[s1, s2]

# 混合信号
A = np.array([[1, 0.5], [0.5, 1]])
X = S.dot(A.T)

# ICA 分离
ica = FastICA(n_components=2)
S_est = ica.fit_transform(X)  # 提取独立成分

# 画图
plt.figure()
plt.plot(S_est)
plt.title('Separated ICA Components')
plt.show()

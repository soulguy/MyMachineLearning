import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#    当参数为None时，生成的数据为随机数据。为常数时，生成恒定数据
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)
print(X)
print(y)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
# clf.coef_代表w数组
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]


wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)
ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()

plt.axis('tight')
plt.show()

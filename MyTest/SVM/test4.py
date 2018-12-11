import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
np.random.seed(0)
# np.random.rand()是产生位于[0,1)之间的随机数 而np.random.randn()是产生服从生态分布的随机数
X = np.r_[np.random.randn(20, 2) - [2,2], np.random.randn(20, 2) + [2,2]]
print("X:",X)
y = [0]*20+[1]*20
clf = svm.SVC(kernel="linear")
clf.fit(X,y)


#define the xx
xx = np.linspace(-10,10)
# get the array w
w = clf.coef_[0]
# the classifier line obey the rules: w_0*x+w_1*y+w_3=0
# the slope
a = - w[0]/w[1]
# the intercept
intercept = clf.intercept_[0]/w[1]
# the supporter line
yy = a*xx-intercept


print("w:",w)
print("a:",a)
print("xx:",xx)
print("yy:",yy)
print("clf.support_vertors:",clf.support_vectors_)

# the nearest two line
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
# the first support verctor -> down
yy_down = yy - np.sqrt(1 + a ** 2) * margin
# the last support vector -> up
yy_up = yy + np.sqrt(1 + a ** 2) * margin

plt.plot(xx,yy,"k-")
plt.plot(xx,yy_down,"k--")
plt.plot(xx,yy_up,"k--")
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
            facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')
plt.legend()
plt.axis("tight")
plt.show()
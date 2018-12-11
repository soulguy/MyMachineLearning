import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
X = np.r_[np.random.randn(20, 2) - [2,2], np.random.randn(20, 2) + [2,2]]
y = [0]*20+[1]*20




clf = svm.SVC(kernel="linear")
clf.fit(X,y)

xx = np.linspace(-10,10)
w = clf.coef_[0]

a = -w[0]/w[1]
# the intercept is clf.intercept_[0]/w[1]
# this is the middlemost line
yy = a*xx-clf.intercept_[0]/w[1]

margin = 1/np.sqrt(np.sum(clf.coef_**2))

yy_down = yy - np.sqrt(1+a**2)*margin

yy_up = yy + np.sqrt(1+a**2)*margin

plt.plot(xx,yy,"k-")
plt.plot(xx,yy_down,"k--")
plt.plot(xx,yy_up,"k--")
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,
            facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X[:,0],X[:,1],c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

plt.axis("tight")
plt.show()



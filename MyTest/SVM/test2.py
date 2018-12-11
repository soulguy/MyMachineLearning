import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
np.random.seed(0)
x = np.sort(5*np.random.rand(40,1),axis=0) #产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列
y = np.sin(x).ravel() #np.sin()输出的是列，和X对应，ravel表示转换成行

# Add noise to targets
y[::5] += 3*(0.5-np.random.rand(8))
print(y)

svr_rbf = SVR(kernel="rbf",C=1e3,gamma=0.1)
svr_lin = SVR(kernel="linear",C=1e3)
svr_poly = SVR(kernel="poly",C=1e3,degree=2)
y_rbf = svr_rbf.fit(x,y).predict(x)
y_lin = svr_lin.fit(x,y).predict(x)
y_poly = svr_poly.fit(x,y).predict(x)

lw = 2
plt.scatter(x,y,color="darkorange",label="data")
plt.plot(x,y_rbf,color="navy",lw=lw,label="RBF model")
plt.plot(x,y_lin,color="c",lw=lw,label="Linear model")
plt.plot(x,y_poly,color="cornflowerblue",lw=lw,label="Polynomial model")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Support Vectot Regression")
plt.legend()
plt.show()


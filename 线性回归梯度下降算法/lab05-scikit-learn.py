#使用scikit-learn的线性回归模型进行线性回归预测


import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def main():
    X_train = np.array([[2104., 5., 1., 45.], [1416., 3., 2., 40.], [852., 2., 1., 35.]])
    y_train = np.array([460., 232., 178.])
    X_features = ['size(sqrt)', 'bedrooms', 'floors', 'age']

    #标准化
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train) #使用的是z-score标准化

    #Create and fit the regression model, 使用的是随机/小批量梯度下降法而不是批量梯度下降法，批量 GD 收敛路径更平滑但每步开销大；SGD 收敛更快（尤其大数据）但轨迹有随机振荡，通常需要更小或衰减的学习率并经常进行多轮 epoch。
    sgdr = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)  #tol表示收敛阈值(cost function的值小于该值则认为达到收敛)，learning_rate选择学习率衰减方式，eta0是初始学习率
    sgdr.fit(X_norm, y_train)

    b_norm = sgdr.intercept_[0]
    w_norm = sgdr.coef_


    #make predictions
    y_pred_sgd = sgdr.predict(X_norm)
    y_pred = np.dot(X_norm, w_norm) + b_norm

    #plot results
    fig,ax=plt.subplots(1,4,figsize=(12,4),sharey=True) #创建画布： 1行4列子图,宽 12 英寸、高 4 英寸,共享y轴
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],y_pred,color='orange', label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()


if __name__ == '__main__':
    main()
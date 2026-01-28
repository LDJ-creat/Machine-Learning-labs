import numpy as np
import matplotlib.pyplot as plt
import math

def compute_model_output(x,w,b):

    return np.dot(x, w) + b


def compute_cost(w,b,x,y):
    # 向量化实现，防止数值溢出并更高效
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)
    residuals = f_wb - y
    cost = np.sum(residuals ** 2) / (2 * m)
    return cost

def compute_gradient(w,b,x,y):
    # 向量化实现：更稳定且更快
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)
    errors = f_wb - y
    dj_dw = np.dot(x.T, errors) / m  #x.T 是转置,将（m,n）变为（n,m）
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,decay=0.0,method='time'):
    J_history=[] # 记录成本函数历史
    p_history=[] # 记录参数更新历史   
    w=w_in
    b=b_in

    for i in range(num_iters):
        dj_dw,dj_db=compute_gradient(w,b,x,y)
        # 计算动态学习率：
        # time-based: alpha / (1 + decay * i) --小幅、线性式衰减
        # exp: alpha * (decay ** i)  (decay 应接近 0.99...) --适合想快速降步长的场景。
        if decay and method == 'exp':
            current_alpha = alpha * (decay ** i)
        else:
            current_alpha = alpha / (1 + decay * i)

        w = w - current_alpha * dj_dw
        b = b - current_alpha * dj_db

        # 保存历史记录
        if i < 100000:      # 防止内存溢出
            J_history.append(compute_cost(w,b,x,y))
            p_history.append((w,b,current_alpha)) 

        # 每10%迭代打印一次状态（为 ndarray 单独格式化）
        if i % math.ceil(num_iters/10) == 0:
            dj_dw_str = np.array2string(dj_dw, formatter={'float_kind':lambda x: f"{x:0.3e}"})
            w_str = np.array2string(w, formatter={'float_kind':lambda x: f"{x:0.3e}"})
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}, dj_dw: {dj_dw_str}, dj_db: {dj_db:0.3e}, alpha: {current_alpha:0.3e}, w: {w_str}, b: {b:0.5e}")
    return w,b,J_history,p_history


def main():
    #生成数据
    X_train = np.array([[2104., 5., 1., 45.], [1416., 3., 2., 40.], [852., 2., 1., 35.]])
    y_train = np.array([460., 232., 178.])

    # 特征标准化（均值归一化），对梯度下降非常重要:
    # 统一尺度：把不同量纲/量级的特征变到相近尺度（均值约为 0、标准差为 1），避免大尺度特征主导梯度。
    # 加速收敛：梯度下降的等高线更接近圆形，能用更合理的学习率，减少振荡、收敛更快。
    # 数值稳定性：减少溢出/NaN、避免梯度爆炸或权重跳变。
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_norm = (X_train - mu) / sigma

    n_features = X_train.shape[1]
    w_init = np.zeros(n_features)
    b_init = 0.0
    iterations = 1000
    alpha = 0.01 # 基础学习率（在标准化后通常可以选得稍大）

    # 训练（示例使用 time-based 衰减）
    w_final, b_final, J_hist, p_hist = gradient_descent(X_norm, y_train, w_init, b_init, alpha, iterations, decay=0.001, method='time')

    # 预测值（用与训练相同的缩放）
    x_test = np.array([1200.0, 3.0, 1.0, 40.0])
    x_test_norm = (x_test - mu) / sigma
    y_pred = compute_model_output(x_test_norm, w_final, b_final)
    print(f"预测结果: {y_pred}")

    #可视化
    plt.plot(J_hist)
    plt.xlabel("迭代次数")
    plt.ylabel("成本函数值")
    plt.title("成本函数值随迭代次数变化")
    plt.show()  

if __name__=="__main__":
    main()
    
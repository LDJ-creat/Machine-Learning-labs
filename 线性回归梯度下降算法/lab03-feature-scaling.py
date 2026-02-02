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
    # --- 三种特征缩放实现并对比 ---
    # 1) 最大值归一化（按列最大值缩放到 [0,1]）
    X_max = np.max(X_train, axis=0)
    X_max_safe = X_max.copy()
    X_max_safe[X_max_safe == 0] = 1.0
    X_maxnorm = X_train / X_max_safe

    # 2) min-max scaling（缩放到 [0,1]）
    X_min = np.min(X_train, axis=0)
    X_range = X_max - X_min
    X_range_safe = X_range.copy()
    X_range_safe[X_range_safe == 0] = 1.0
    X_minmax = (X_train - X_min) / X_range_safe

    # 3) z-score normalization（均值为0，标准差为1）
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe == 0] = 1.0
    X_zscore = (X_train - mu) / sigma_safe

    n_features = X_train.shape[1]
    iterations = 1000
    alpha = 0.01

    scalings = {
        'max': {'X': X_maxnorm, 'params': (X_max_safe,)},
        'minmax': {'X': X_minmax, 'params': (X_min, X_range_safe)},
        'zscore': {'X': X_zscore, 'params': (mu, sigma_safe)},
    }

    results = {}

    # 测试样本（对比时需使用相同的原始样本并按相同方式缩放）
    x_test = np.array([1200.0, 3.0, 1.0, 40.0])

    for name, info in scalings.items():
        X_scaled = info['X']
        w_init = np.zeros(n_features)
        b_init = 0.0

        w_final, b_final, J_hist, p_hist = gradient_descent(X_scaled, y_train, w_init, b_init,
                                                            alpha, iterations, decay=0.001, method='time')

        # 对测试样本做相同的缩放
        if name == 'max':
            x_test_scaled = x_test / info['params'][0]
        elif name == 'minmax':
            X_min_val, X_range_val = info['params']
            x_test_scaled = (x_test - X_min_val) / X_range_val
        else:  # zscore
            mu_val, sigma_val = info['params']
            x_test_scaled = (x_test - mu_val) / sigma_val

        y_pred = compute_model_output(x_test_scaled, w_final, b_final)

        results[name] = {
            'w': w_final,
            'b': b_final,
            'J_hist': J_hist,
            'y_pred': y_pred,
            'final_cost': J_hist[-1] if len(J_hist) > 0 else None,
        }

        print(f"方法 {name}: 最终成本 {results[name]['final_cost']:0.4f}, 预测值 {y_pred:0.4f}")

    # 绘制三种方法的学习曲线对比
    plt.figure()
    for name, res in results.items():
        plt.plot(res['J_hist'], label=name)
    plt.xlabel("迭代次数")
    plt.ylabel("成本函数值")
    plt.title("三种特征缩放的成本函数收敛对比")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
    
import numpy as np
import matplotlib.pyplot as plt
import math

def compute_model_output(x,w,b):

    return np.dot(x, w) + b


def compute_cost(w,b,x,y):
    m=x.shape[0]
    f_wb=compute_model_output(x,w,b)
    cost = np.sum((f_wb-y)**2)/(2*m)    
    return cost

def compute_gradient(w,b,x,y):
    m=x.shape[0]
    f_wb=compute_model_output(x,w,b)
    dj_dw=np.sum((f_wb-y)*x)/m
    dj_db=np.sum(f_wb-y)/m
    return dj_dw,dj_db

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

        # 每10%迭代打印一次状态
        if i % math.ceil(num_iters/10) == 0:    
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}, dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}, alpha: {current_alpha:0.3e}, w: {w:0.3e}, b: {b:0.5e}")    
    return w,b,J_history,p_history


def main():
    #生成数据
    x_train=np.array([1.0,2.0,3.0,4.0,5.0])
    y_train = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])  # 目标值（如：千美元）
    w_init=0.0
    b_init=0.0
    iterations=1000
    alpha=0.01 #学习率

    # 示例：启用 time-based 衰减（decay>0），或 method='exp' 使用指数衰减
    # decay=0.0 保持原来固定学习率行为
    w_final,b_final,J_hist,p_hist=gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations,decay=0.001,method='time')

    #预测值
    x_test=np.array([6.0,7.0,8.0])
    y_pred=compute_model_output(x_test,w_final,b_final)
    print(f"预测结果: {y_pred}")

    #可视化
    plt.plot(J_hist)
    plt.xlabel("迭代次数")
    plt.ylabel("成本函数值")
    plt.title("成本函数值随迭代次数变化")
    plt.show()  

if __name__=="__main__":
    main()
    
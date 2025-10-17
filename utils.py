import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy import stats


def generate_disturbances(dis, P, N, gaussian_index):
    disturbances = 0
    if dis == "uniform":
        a = -np.sqrt(3)
        b = np.sqrt(3)
        # # disturbance = np.random.uniform(a, b, size=(N, P))
        # disturbances = np.random.uniform(a, b, size=(N, P))
        # alphas = np.random.uniform(1.5, 2, P)
        # betas = np.random.uniform(2, 10, P)
        disturbances = np.random.uniform(a, b, size=(P, N))
    elif dis == "exponential":
        scale = 1.0  # λ = 1
        disturbances = np.random.exponential(scale, size=(P, N)) - scale
        # disturbances = np.random.exponential(scale, size=(P, N))
    elif dis == "gamma":
        # shape = 1
        # scale = 2
        # disturbances = np.random.gamma(shape, scale, size=(N, P)) - 2.0
        # shapes = np.random.uniform(0.1, 1, P)
        # scales = np.random.uniform(0.1, 0.5, P)
        # shapes = np.random.uniform(1.0, 3.0, P)
        shapes = np.random.uniform(2.0, 3.0, P)
        # scales = np.random.uniform(0.5, 0.5, P)
        scales = np.random.uniform(0.3, 0.5, P)
        # scales = np.random.uniform(0.1, 0.5, P)
        scales = 1 / np.sqrt(shapes)
        disturbances = np.array([np.random.gamma(shape, scale, N) - shape * scale for shape, scale in zip(shapes, scales)])
    elif dis == "beta":
        alphas = np.random.uniform(1.5, 2, P)
        betas = np.random.uniform(2, 10, P)
        disturbances = np.array([np.random.beta(alpha, beta, N) - alpha / (alpha + beta) for alpha, beta in zip(alphas, betas)])
    elif dis == "lognormal":
        # mus = np.random.uniform(-2, -0.5, P)
        # sigmas = np.random.uniform(0.1, 0.4, P)
        # mus = -0.5
        # sigmas = 1
        # mus = np.random.uniform(-0.5, -0.5, P)
        # mus = np.random.uniform(0.2, 0.3, P)
        # sigmas = np.random.uniform(0.2, 0.6, P)
        # mus = np.random.uniform(-0.5, -0.5, P)
        # sigmas = np.random.uniform(0.5, 0.5, P)
        # mus    = np.random.uniform(-2.0, -0.5, size=P)
        # sigmas = np.random.uniform( 0.2,  1.0, size=P)
        # mus = np.random.uniform(0.0, 1.0, P)
        # sigmas = np.random.uniform(0.25, 1.0, P)
        # mus = np.random.uniform(0.0, 0.0, P)
        # sigmas = np.random.uniform(0.25, 0.25, P)
        mus = np.random.uniform(-1.1, -1.1, P)
        sigmas = np.random.uniform(0.8, 0.8, P)
        disturbances = np.array([np.random.lognormal(mu, sigma, N) - np.exp(mu + sigma ** 2 / 2) for mu, sigma in zip(mus, sigmas)])
    for i in gaussian_index:
        print(i)
        disturbances[i, :] = np.random.normal(0, 1, N)
    return disturbances


def generate_coefficient_matrix(matrix, hidden_num):
    num = matrix.shape
    # coef_matrix = np.where(matrix > 0, np.random.uniform(-0.5, 0.5), matrix)
    coef_matrix = np.zeros_like(matrix, dtype=float)
    # print(coef_matrix)
    for i in range(num[0]):
        for j in range(num[1]):
            if matrix[i][j] > 0:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    # coef_matrix[i][j] = np.random.uniform(1.5, 3.0)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 1.5)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 3.0)
                # coef_matrix[i][j] = np.round(np.random.uniform(1.5, 5.0), 3)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 0.9)
                    # coef_matrix[i][j] = np.random.uniform(1.1, 2.0)
                    coef_matrix[i][j] = np.random.uniform(1.1, 1.5) # <-
                    # coef_matrix[i][j] = np.random.uniform(1.5, 2.0)
                else:
                    # coef_matrix[i][j] = np.random.uniform(1.5, 3.0) * (-1)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 1.5) * (-1)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 3.0) * (-1)
                    # coef_matrix[i][j] = (-1.0) * np.random.uniform(0.5, 0.9)
                    # coef_matrix[i][j] = np.random.uniform(0.5, 0.9)
                    # coef_matrix[i][j] = -1.0 * np.random.uniform(1.1, 2.0)
                    # coef_matrix[i][j] = np.random.uniform(1.5, 2.0)
                    # coef_matrix[i][j] = np.random.uniform(0.4, 0.9)
                    # coef_matrix[i][j] = np.random.uniform(1.1, 1.4) * (-1.0) # <-
                    # coef_matrix[i][j] = np.random.uniform(1.1, 2.0) * (-1.0) # <-
                    # coef_matrix[i][j] = np.random.uniform(1.1, 2.0) 
                    coef_matrix[i][j] = np.random.uniform(1.1, 1.5) # <-
                # coef_matrix[i][j] = 1
    # coef_matrix = np.round(coef_matrix)
    # for i in range(hidden_num):
    #     for j in range(hidden_num):
    #         if coef_matrix[i][j]:
    #             coef_matrix[i][j] = np.round(coef_matrix[i][j])
    for i in range(num[0] - hidden_num):
        for j in range(num[0] - hidden_num):
            if matrix[i + hidden_num][j + hidden_num] != 0.0:
                coef_matrix[i + hidden_num][j + hidden_num] = np.random.uniform(0.5, 0.9)
    for i in range(hidden_num):
        for j in range(num[0] - hidden_num):
            if coef_matrix[j + hidden_num][i]!=0:
                coef_matrix[j + hidden_num][i] = 1.0
                break
    return coef_matrix


def generate_data(disturbance, coefficient_matrix, hidden_num):
    num = coefficient_matrix.shape
    A = np.linalg.inv(np.eye(num[0]) - coefficient_matrix)
    # x = np.dot(A, disturbance).T
    x = (A @ disturbance).T
    # print(x.shape)
    x_hidden = x[:, :hidden_num]
    x_observed = x[:, hidden_num:]
    return x, x_hidden, x_observed, A, A[:hidden_num, :], A[hidden_num:, :]


def set_partitions(set_):
    """
    递归生成集合 set_ 的所有划分，每个划分表示为块（列表）的列表。
    例如：set_partitions([0, 1]) 会产生 [[0, 1]] 和 [[0], [1]]
    """
    if len(set_) == 1:
        yield [set_]
        return
    first = set_[0]
    # 对剩下的元素递归求划分
    for partition in set_partitions(set_[1:]):
        # 尝试将 first 插入 partition 中已有的每个块
        for i in range(len(partition)):
            new_partition = []
            for j, block in enumerate(partition):
                if i == j:
                    new_partition.append([first] + block)
                else:
                    new_partition.append(block)
            yield new_partition
        # 也可以让 first 单独作为一块
        yield [[first]] + partition


def cumulant(data, indices):
    """
    计算 data 中对应 columns (indices) 的高阶交叉 cumulant.

    参数:
      data: numpy 数组，形状为 (n_samples, n_features)，每一行为一个样本；
      indices: 一个列表，例如 [0,1,2] 表示计算第 1,2,3 列（变量）的联合 cumulant.

    根据公式:
      cumulant(X_i1,...,X_in) = sum_{π in partitions} (|π|-1)! * (-1)^(|π|-1) * ∏_{B in π} E[∏_{j in B} X_{ij}]
    """
    cumulant_value = 0.0
    indices_list = list(indices)

    # print(data.shape)
    # print(indices)

    # 遍历所有的划分
    for partition in set_partitions(indices_list):
        # 对每个划分，计算每个块的样本矩：E[∏_{j in block} X_j]
        prod_moment = 1.0
        for block in partition:
            # axis=1 表示对每个样本计算乘积，再对所有样本求均值
            temp = np.prod(data[:, block], axis=1)
            # print(temp.shape)
            moment = np.mean(np.prod(data[:, block], axis=1))
            prod_moment *= moment
        # 权重为 (|partition|-1)! * (-1)^(|partition|-1)
        weight = math.factorial(len(partition) - 1) * ((-1) ** (len(partition) - 1))
        cumulant_value += weight * prod_moment
    return cumulant_value


def draw(adj_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos,
        with_labels=True,          # 显示节点标签
        node_color='lightblue',      # 节点颜色
        edge_color='gray',           # 边颜色
        arrows=True,                 # 显示箭头
        arrowstyle='<|-',           # 箭头样式
        arrowsize=12)                # 箭头大小
    plt.show()


# def get_all 

# def matrix_rank_svd(matrix, tol=1e-10):
#     # 使用 SVD 分解矩阵，返回 U、奇异值数组 s 和 V 的转置
#     U, s, Vh = np.linalg.svd(matrix)
#     # 将大于容差的奇异值计数，即为矩阵的秩
#     rank = sum(s > tol)
#     return rank

def matrix_rank_svd(matrix, sample_size, latent_num):
    # 使用 SVD 分解矩阵，返回 U、奇异值数组 s 和 V 的转置
    U, s, Vh = np.linalg.svd(matrix)
    print("singular values: ", s)
    # 将大于容差的奇异值计数，即为矩阵的秩
    # tol = 1e-6
    # rank = sum(s > tol)
    flag = False

    tol = 0.008 * float(sample_size)**(-0.02)

    print("threshold: ", tol)
    print("ratio: ", s[latent_num + 2 - 1] / s[0])

    if s[latent_num + 2 - 1] / s[0] <= tol:
        flag = True
    # if s[latent_num + 2 - 1] <= tol:
    #     flag = True
    return flag


def make_cumulant_tensor(X, l_num):
    k_1 = l_num + 2
    x =  int(np.ceil(1/2 * (-3 + np.sqrt(8*l_num +17)))) # k_2 - k_1
    k_2 = k_1 + x

    row_num = int((k_2-k_1+1)*(k_2-k_1+2)/2)
    col_num = k_1 
    A_1_to_2 = np.zeros(shape=(row_num, col_num), dtype=float)
    A_2_to_1 = np.zeros(shape=(row_num, col_num), dtype=float)

    row_index = 0
    # print("x= ", x)
    # print("row_num= ", row_num)
    # print("col_num= ", col_num)
    for i in range(x+1):
        # print("i= ", i)
        temp_col_indices = [0]*(k_1+i)
        temp_col_indices_inverse = [1]*(k_1+i)
        for j in range(i+1): # i次数的时候的行数控制
            if j == 0:
                for col_index in range(col_num):
                    # print("axis: (", row_index, ", ", col_index, ")")
                    if col_index == 0:
                        # temp_col_indices = temp_col_indices
                        # print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
                    else:
                        temp_col_indices[col_index] = 1
                        temp_col_indices_inverse[col_index] = 0
                        # print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
            else:
                for col_index in range(col_num):
                    # print("axis: (", row_index, ", ", col_index, ")")
                    # print(temp_col_indices)
                    if col_index != col_num-1:
                        # print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = A_1_to_2[row_index - 1][col_index + 1]
                        A_2_to_1[row_index][col_index] = A_2_to_1[row_index - 1][col_index + 1]
                    else:
                        temp_col_indices[col_num + j - 1] = 1
                        temp_col_indices_inverse[col_num + j - 1] = 0
                        # print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
            row_index += 1
    return A_1_to_2, A_2_to_1



def is_correct_latent_num_pairwise(X, l_num):
    sample_size = X.shape[0]
    print("sample size: ", sample_size)
    # l_num = 0
    k_1 = l_num + 2
    x =  int(np.ceil(1/2 * (-3 + np.sqrt(8*l_num +17)))) # k_2 - k_1
    k_2 = k_1 + x

    row_num = int((k_2-k_1+1)*(k_2-k_1+2)/2)
    col_num = k_1 
    # A_1_to_2 = list()
    # A_2_to_1 = list()
    A_1_to_2 = np.zeros(shape=(row_num, col_num), dtype=float)
    A_2_to_1 = np.zeros(shape=(row_num, col_num), dtype=float)

    row_index = 0
    print("x= ", x)
    print("row_num= ", row_num)
    print("col_num= ", col_num)
    for i in range(x+1):
        print("i= ", i)
        temp_col_indices = [0]*(k_1+i)
        temp_col_indices_inverse = [1]*(k_1+i)
        for j in range(i+1):
            if j == 0:
                for col_index in range(col_num):
                    print("axis: (", row_index, ", ", col_index, ")")
                    if col_index == 0:
                        # temp_col_indices = temp_col_indices
                        print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
                    else:
                        temp_col_indices[col_index] = 1
                        temp_col_indices_inverse[col_index] = 0
                        print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
            else:
                for col_index in range(col_num):
                    print("axis: (", row_index, ", ", col_index, ")")
                    # print(temp_col_indices)
                    if col_index != col_num-1:
                        print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = A_1_to_2[row_index - 1][col_index + 1]
                        A_2_to_1[row_index][col_index] = A_2_to_1[row_index - 1][col_index + 1]
                    else:
                        temp_col_indices[col_num + j - 1] = 1
                        temp_col_indices_inverse[col_num + j - 1] = 0
                        print(temp_col_indices)
                        A_1_to_2[row_index][col_index] = cumulant(X, temp_col_indices)
                        A_2_to_1[row_index][col_index] = cumulant(X, temp_col_indices_inverse)
            row_index += 1

    # temp = list()
    print("the size of A matrix is ", A_1_to_2.shape)
    flag_1_to_2 = matrix_rank_svd(A_1_to_2, sample_size, l_num)
    flag_2_to_1 = matrix_rank_svd(A_2_to_1, sample_size, l_num)
    print("2->3: ", flag_1_to_2)
    print("3->2: ", flag_2_to_1)
        
    # 如果返回的值为1，说明方向是1 -> 2，且latent的数量为l_num + 1
    # 如果返回的值为2，说明方向是2 -> 1，且latent的数量为l_num + 1
    # 如果返回的值为3，说明2 1之间没有有向边，且latent的数量为l_num + 1
    # 如果返回的值为0，说明latent的数量比l_num + 1 大
    if flag_1_to_2 and not flag_2_to_1:
        return 1
    elif flag_2_to_1 and not flag_1_to_2:
        return 2
    elif flag_1_to_2 and flag_2_to_1:
        return 3
    else:
        return 0

        
def determine_latent_number(X):
    latent_num = 0
    for l_num in range(4):
        print("the number of latent variables is ", l_num , " now.")
        if not is_correct_latent_num_pairwise(X, l_num):
            pass
        else:
            latent_num = l_num
            return latent_num
    return -1
    # for l in 
    # for 

def test_one_confounder(X_1, X_2):
    # indices_33 = [0, 0, 0, 1, 1, 1]
    # indices_42 = [0, 0, 0, 0, 1, 1]
    # indices_24 = [0, 0, 1, 1, 1, 1]
    # indices_33 = [0, 0, 1, 1]
    # indices_42 = [0, 1, 1, 1]
    # indices_24 = [0, 0, 0, 1]
    # cum_33 = cumulant(np.array([X_1, X_2]), indices_33)
    # cum_42 = cumulant(np.array([X_1, X_2]), indices_42)
    # cum_24 = cumulant(np.array([X_1, X_2]), indices_24)

    # print("cum_33: ", cum_33)
    # print("cum_42: ", cum_42)
    # print("cum_24: ", cum_24)
    # print(cum_33 * cum_33)
    # print(cum_42 * cum_24)
    # print("div: ", (cum_33*cum_33 / (cum_42*cum_24)))
    # print("sub: ", np.fabs(1 - (cum_33*cum_33 / (cum_42*cum_24))))
    
    # if np.fabs(1 - (cum_33*cum_33 / (cum_42*cum_24)))< 10e-4:
        
    #     return True
    # else:
    #     return False
    l_num = 1
    data = np.array([X_1, X_2]).T
    # print("data shape: ", data.shape)
    # print(data)
    A_1_to_2, A_2_to_1 = make_cumulant_tensor(data, l_num)
    U, s_1, Vh = np.linalg.svd(A_1_to_2)
    U, s_2, Vh = np.linalg.svd(A_2_to_1)
    print(s_1)
    print(s_2)
    print(s_1[l_num + 2 - 1] / s_1[0])
    print(s_2[l_num + 2 - 1] / s_2[0])
    if(s_1[l_num + 2 - 1] / s_1[0])> 1e-4 and (s_2[l_num + 2 - 1] / s_2[0])> 0.06:
        return False
    else:
        return True


def test_one_confounder_sixth(X_1, X_2):
    data = np.array([X_1, X_2]).T
    indices_33 = [0, 0, 0, 1, 1, 1]
    indices_42 = [0, 0, 0, 0, 1, 1]
    indices_24 = [0, 0, 1, 1, 1, 1]
    # indices_33 = [0, 0, 1, 1]
    # indices_42 = [0, 1, 1, 1]
    # indices_24 = [0, 0, 0, 1]
    cum_33 = cumulant(data, indices_33)
    cum_42 = cumulant(data, indices_42)
    cum_24 = cumulant(data, indices_24)

    print("cum_33: ", cum_33)
    print("cum_42: ", cum_42)
    print("cum_24: ", cum_24)
    print(cum_33 * cum_33)
    print(cum_42 * cum_24)
    print("sub: ", cum_33*cum_33 - (cum_42*cum_24))
    print("div: ", (cum_33*cum_33 / (cum_42*cum_24)))
    print("sub: ", np.fabs(1 - (cum_33*cum_33 / (cum_42*cum_24))))
    
    if np.fabs(cum_33*cum_33 - (cum_42*cum_24))< 1e-6:  
        return True
    else:
        return False
    

# def estimate_b(A_1_to_2)

def evaluate_one(G_est, ordered_cluster_est, observed_edge_est, G_true, ordered_cluster_true, observed_edge_true, is_proposed):
    """
    评估估计的图结构与真实图结构之间的差异。
    返回 True 表示估计正确，False 表示估计错误。
    """
    is_cluster_correct = False
    cluster_correct_count_est = 0
    cluster_add_count_est = 0
    cluster_miss_count_est = 0
    latent_count_est = len(ordered_cluster_est)
    latent_count_true = len(ordered_cluster_true)
    
    for c in ordered_cluster_est:
        sorted(c)
    ordered_cluster_est = sorted(ordered_cluster_est, key=lambda x: x[0])
    
    for c in ordered_cluster_true:
        sorted(c)
    ordered_cluster_true = sorted(ordered_cluster_true, key=lambda x: x[0])

    for c in ordered_cluster_est:
        print("c: ", c)
        print("c: ", ordered_cluster_true)
        if c in ordered_cluster_true:
            cluster_correct_count_est += 1   
        else:
            cluster_add_count_est += 1
    
    cluster_miss_count_est = len(ordered_cluster_true) - cluster_correct_count_est

    print(cluster_correct_count_est)
    print(cluster_add_count_est)
    print(cluster_miss_count_est)
    
    cluster_PRE = cluster_correct_count_est / (cluster_correct_count_est + cluster_add_count_est) if (cluster_correct_count_est + cluster_add_count_est) > 0 else 0
    cluster_REC = cluster_correct_count_est / (cluster_correct_count_est + cluster_miss_count_est) if (cluster_correct_count_est + cluster_miss_count_est) > 0 else 0
    cluster_F1 = 2 * cluster_PRE * cluster_REC / (cluster_PRE + cluster_REC) if (cluster_PRE + cluster_REC) > 0 else 0
    
    if cluster_correct_count_est == latent_count_true:
        is_cluster_correct = True
    
    edge_correct_count_est = 0
    edge_add_count_est = 0
    edge_miss_count_est = 0
    error_squared_est = 0.0
    edge_num_est = 0

    observed_edge_correct_count_est = 0
    observed_edge_add_count_est = 0
    observed_edge_miss_count_est = 0
    error_squared_est = 0.0
    observed_edge_num_est = 0

    if is_cluster_correct:
        if is_proposed:
            for i in range(len(G_est)):
                for j in range(len(G_est)):
                    if G_est[i][j] != 0.0 and G_true[i][j] != 0.0:
                        edge_correct_count_est += 1
                        # error_squared_est += (G_est[i][j] - G_true[i][j]) ** 2
                        # edge_num_est += 1
                    elif G_est[i][j] != 0.0 and G_true[i][j] == 0.0:
                        edge_add_count_est += 1
                        # error_squared_est += (G_est[i][j] - G_true[i][j]) ** 2
                        # edge_num_est += 1
                    elif G_est[i][j] == 0.0 and G_true[i][j] != 0.0:
                        edge_miss_count_est += 1
            error_squared_est = np.mean((G_est - G_true) ** 2)

            for k, v in observed_edge_true.items():
                if k not in observed_edge_est.keys():
                    observed_edge_miss_count_est += len(v)
                else:
                    for i in v:
                        if i in observed_edge_est[k]:
                            observed_edge_correct_count_est += 1
            
            for k, v in observed_edge_est.items():
                if k not in observed_edge_true:
                    observed_edge_add_count_est += len(v)
                else:
                    for i in v:
                        if i not in observed_edge_true[k]:
                            observed_edge_add_count_est += 1
        else:
            for i in range(len(G_est)):
                for j in range(len(G_est)):
                    if G_est[i][j] != 0.0 and G_true[i][j] != 0.0:
                        edge_correct_count_est += 1
                    elif G_est[i][j] != 0.0 and G_true[i][j] == 0.0:
                        edge_add_count_est += 1
                    elif G_est[i][j] == 0.0 and G_true[i][j] != 0.0:
                        edge_miss_count_est += 1
    
    print("edge_correct_count_est: ", edge_correct_count_est)
    print("edge_add_count_est: ", edge_add_count_est)
    print("edge_miss_count_est: ", edge_miss_count_est) 
    print("error_squared_est: ", error_squared_est)
    print("edge_num_est: ", edge_num_est)
    edge_PRE = edge_correct_count_est / (edge_correct_count_est + edge_add_count_est) if (edge_correct_count_est + edge_add_count_est) > 0 else 0.0
    edge_REC = edge_correct_count_est / (edge_correct_count_est + edge_miss_count_est) if (edge_correct_count_est + edge_miss_count_est) > 0 else 0.0
    edge_F1 = 2 * edge_PRE * edge_REC / (edge_PRE + edge_REC) if (edge_PRE + edge_REC) > 0 else 0.0

    MSE = error_squared_est
    RMSE = np.sqrt(MSE)
    return [is_cluster_correct, cluster_PRE, cluster_REC, cluster_F1, edge_PRE, edge_REC, edge_F1, RMSE, observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est]

    # return is_cluster_correct, right_cluster_count_est, latent_count_est, latent_count_true, edge_correct_count_est, edge_add_count_est, edge_miss_count_est, error_squared_est
    
def evaluate_repeat(res_list):
#     """
#     重复评估函数，适用于多次实验的结果评估。
#     返回一个包含所有评估结果的列表。
#     """
    
    # print(res_list)
    is_cluster_correct_count = np.sum([res[0] for res in res_list])
    cluster_PRE = np.mean([res[1] for res in res_list])
    cluster_REC = np.mean([res[2] for res in res_list])
    cluster_F1 = np.mean([res[3] for res in res_list])
    
    edge_PRE = np.mean([res[4] for res in res_list if res[4] != 0]) if len([res[4] for res in res_list if res[4] != 0]) != 0 else 0
    edge_REC = np.mean([res[5] for res in res_list if res[5] != 0]) if len([res[5] for res in res_list if res[5] != 0]) != 0 else 0
    edge_F1 = np.mean([res[6] for res in res_list if res[6] != 0]) if len([res[6] for res in res_list if res[6] != 0]) != 0 else 0
    RMSE_mean = np.mean([res[7] for res in res_list if res[7] != 0]) if len([res[7] for res in res_list if res[7] != 0]) != 0 else float('inf')
    RMSE_std = np.std([res[7] for res in res_list if res[7] != 0])  if len([res[7] for res in res_list if res[7] != 0]) != 0 else float('inf')

    observed_edge_correct = np.sum([res[8] for res in res_list])
    observed_edge_add = np.sum([res[9] for res in res_list])
    observed_edge_miss = np.sum([res[10] for res in res_list])

    observed_edge_PRE = observed_edge_correct / (observed_edge_correct + observed_edge_add) if (observed_edge_correct + observed_edge_add) > 0 else 0.0
    observed_edge_REC = observed_edge_correct / (observed_edge_correct + observed_edge_miss) if (observed_edge_correct + observed_edge_miss) > 0 else 0.0
    observed_edge_F1 = 2 * observed_edge_PRE * observed_edge_REC / (observed_edge_PRE + observed_edge_REC) if (observed_edge_PRE + observed_edge_REC) > 0 else 0.0    
    return is_cluster_correct_count, cluster_PRE, cluster_REC, cluster_F1, edge_PRE, edge_REC, edge_F1, observed_edge_PRE, observed_edge_REC, observed_edge_F1, RMSE_mean, RMSE_std
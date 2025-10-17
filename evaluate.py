import numpy as np

def evaluate_one(G_est, ordered_cluster_est, observed_edge_est, G_true, ordered_cluster_true, observed_edge_true, is_proposed):
    """
    评估估计的图结构与真实图结构之间的差异。
    返回 True 表示估计正确，False 表示估计错误。
    """
    # 聚类评估
    cluster_correct_count_est, cluster_add_count_est, cluster_miss_count_est = _evaluate_clusters(
        ordered_cluster_est, ordered_cluster_true
    )
    
    latent_count_true = len(ordered_cluster_true)
    # is_cluster_correct = (cluster_correct_count_est == latent_count_true)
    is_cluster_correct = (cluster_correct_count_est == latent_count_true and cluster_add_count_est == 0 and cluster_miss_count_est == 0)  # 至少正确识别一个聚类
    
    # 计算聚类指标
    cluster_PRE, cluster_REC, cluster_F1 = _calculate_metrics(
        cluster_correct_count_est, cluster_add_count_est, cluster_miss_count_est
    )
    
    is_latent_edge_correct = False
    is_observed_edge_correct = False
    # 边评估（只有聚类正确时才评估边）
    if is_cluster_correct:
        edge_correct_count_est, edge_add_count_est, edge_miss_count_est, error_squared_est = _evaluate_edges(
            G_est, G_true
        )
        # 边的正确性判断，当推测图与真实图的结构完全一致时，is_latent_edge_correct 为 True
        if len(G_true) == 1:
            is_latent_edge_correct = True
        else:
            if edge_add_count_est == 0 and edge_miss_count_est == 0:
                is_latent_edge_correct = True   
    
        if is_proposed:
        # if True:
            observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est = _evaluate_observed_edges(
                observed_edge_est, observed_edge_true
            )
            if observed_edge_add_count_est == 0 and observed_edge_miss_count_est == 0:
                is_observed_edge_correct = True
        else:
            # observed_edge_correct_count_est = observed_edge_add_count_est = observed_edge_miss_count_est = 0
            observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est = _evaluate_observed_edges(
                observed_edge_est, observed_edge_true
            )
            if observed_edge_add_count_est == 0 and observed_edge_miss_count_est == 0:
                is_observed_edge_correct = True
            error_squared_est = 0.0
    else:
        edge_correct_count_est = edge_add_count_est = edge_miss_count_est = 0
        error_squared_est = 0.0
        observed_edge_correct_count_est = observed_edge_add_count_est = observed_edge_miss_count_est = 0
    
    # 计算边指标
    edge_PRE, edge_REC, edge_F1 = _calculate_metrics(
        edge_correct_count_est, edge_add_count_est, edge_miss_count_est
    )

    # 计算可观测边
    observed_edge_PRE, observed_edge_REC, observed_edge_F1 = _calculate_metrics(
        observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est
    )
    
    RMSE = np.sqrt(error_squared_est)

    if is_latent_edge_correct:
        error_squared_est_correct = np.mean((G_est - G_true) ** 2)
    else:
        error_squared_est_correct = 0.0
    RMSE_correct = np.sqrt(error_squared_est_correct)
    
    # 图完全正确的时候的count
    is_totally_correct = is_latent_edge_correct and is_observed_edge_correct

    print(f"Cluster: correct={cluster_correct_count_est}, add={cluster_add_count_est}, miss={cluster_miss_count_est}")
    print(f"Edge: correct={edge_correct_count_est}, add={edge_add_count_est}, miss={edge_miss_count_est}")
    print(f"Error: MSE={error_squared_est}, RMSE={RMSE}")
    
    # return [
    #     is_cluster_correct, cluster_PRE, cluster_REC, cluster_F1,
    #     edge_PRE, edge_REC, edge_F1, RMSE,
    #     observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est
    # ]
    print(observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est)
    return [
        is_cluster_correct, is_latent_edge_correct, is_totally_correct,
        cluster_PRE, cluster_REC, cluster_F1,
        edge_PRE, edge_REC, edge_F1, RMSE,
        # observed_edge_correct_count_est, observed_edge_add_count_est, observed_edge_miss_count_est
        observed_edge_PRE, observed_edge_REC, observed_edge_F1,
        RMSE_correct,
        is_observed_edge_correct
    ]


def _evaluate_clusters(ordered_cluster_est, ordered_cluster_true):
    """评估聚类结果"""
    # 标准化聚类格式（排序但不修改原始数据）
    cluster_est_sorted = [sorted(c) for c in ordered_cluster_est]
    cluster_true_sorted = [sorted(c) for c in ordered_cluster_true]
    
    # 按第一个元素排序
    cluster_est_sorted.sort(key=lambda x: x[0])
    cluster_true_sorted.sort(key=lambda x: x[0])
    
    print("Estimated clusters:", cluster_est_sorted)
    print("True clusters:", cluster_true_sorted)
    
    # 计算正确识别的聚类数量
    cluster_correct_count = sum(1 for c in cluster_est_sorted if c in cluster_true_sorted)
    cluster_add_count = len(cluster_est_sorted) - cluster_correct_count
    cluster_miss_count = len(cluster_true_sorted) - cluster_correct_count
    
    return cluster_correct_count, cluster_add_count, cluster_miss_count


def _evaluate_edges(G_est, G_true):
    """评估边结构"""
    # 使用向量化操作提高效率
    print("Estimated graph:", G_est)
    print("True graph:", G_true)
    est_nonzero = (G_est != 0.0)
    true_nonzero = (G_true != 0.0)
    
    edge_correct_count = np.sum(est_nonzero & true_nonzero)
    edge_add_count = np.sum(est_nonzero & ~true_nonzero)
    edge_miss_count = np.sum(~est_nonzero & true_nonzero)
    
    # 计算均方误差
    error_squared = np.mean((G_est - G_true) ** 2)
    
    return edge_correct_count, edge_add_count, edge_miss_count, error_squared


def _evaluate_observed_edges(observed_edge_est, observed_edge_true):
    """评估观测变量间的边"""
    observed_edge_correct_count = 0
    observed_edge_add_count = 0
    observed_edge_miss_count = 0
    
    # 计算漏识别的边
    for k, v in observed_edge_true.items():
        if k not in observed_edge_est:
            observed_edge_miss_count += len(v)
        else:
            est_edges = set(observed_edge_est[k])
            for edge in v:
                if edge in est_edges:
                    observed_edge_correct_count += 1
                else:
                    observed_edge_miss_count += 1
    
    # 计算误识别的边
    for k, v in observed_edge_est.items():
        if k not in observed_edge_true:
            observed_edge_add_count += len(v)
        else:
            true_edges = set(observed_edge_true[k])
            for edge in v:
                if edge not in true_edges:
                    observed_edge_add_count += 1
    
    return observed_edge_correct_count, observed_edge_add_count, observed_edge_miss_count


def _calculate_metrics(correct_count, add_count, miss_count):
    """计算精确率、召回率和F1分数"""
    precision = correct_count / (correct_count + add_count) if (correct_count + add_count) > 0 else 0.0
    recall = correct_count / (correct_count + miss_count) if (correct_count + miss_count) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def evaluate_repeat(res_list):
    """
    重复评估函数，适用于多次实验的结果评估。
    返回一个包含所有评估结果的列表。
    """
    if not res_list:
        return [0] * 17
    
    # 提取各项指标，避免重复列表推导
    is_cluster_correct_list = [res[0] for res in res_list]
    is_latent_edge_correct_list = [res[1] for res in res_list]
    is_totally_correct_list = [res[2] for res in res_list]
    cluster_metrics = [[res[3], res[4], res[5]] for res in res_list]
    edge_metrics = [[res[6], res[7], res[8]] for res in res_list]
    # edge_metrics = [[res[6], res[7], res[8]] for res in res_list]
    rmse_list = [res[9] for res in res_list]
    observed_edge_metrics = [[res[10], res[11], res[12]] for res in res_list]
    rmse_correct_list = [res[13] for res in res_list]
    is_observed_edge_correct_list = [res[14] for res in res_list]

    print(is_cluster_correct_list)
    # 基本统计
    is_cluster_correct_count = sum(is_cluster_correct_list)
    is_latent_edge_correct_count = sum(is_latent_edge_correct_list)
    is_totally_correct_count = sum(is_totally_correct_list)
    is_observed_edge_correct_count = sum(is_observed_edge_correct_list)

    # 聚类指标平均值
    cluster_PRE = np.mean([metrics[0] for metrics in cluster_metrics])
    cluster_REC = np.mean([metrics[1] for metrics in cluster_metrics])
    cluster_F1 = np.mean([metrics[2] for metrics in cluster_metrics])
    
    # 边指标平均值（只考虑非零值）
    # edge_PRE, edge_REC, edge_F1 = _calculate_filtered_means(edge_metrics)
    a = [edge_metrics[i][0] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]
    print(a)
    edge_PRE = np.mean([edge_metrics[i][0] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) if len([edge_metrics[i][0] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0
    edge_REC = np.mean([edge_metrics[i][1] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) if len([edge_metrics[i][1] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0
    edge_F1 = np.mean([edge_metrics[i][2] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) if len([edge_metrics[i][2] for i in range(len(edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0


    # RMSE统计
    valid_rmse = [rmse for rmse in rmse_list if rmse != 0]
    if valid_rmse:
        RMSE_mean = np.mean(valid_rmse)
        RMSE_std = np.std(valid_rmse)
    else:
        RMSE_mean = RMSE_std = float('inf')

    valid_rmse_correct = [rmse for rmse in rmse_correct_list if rmse != 0]
    if valid_rmse_correct:
        RMSE_correct_mean = np.mean(valid_rmse_correct)
        RMSE_correct_std = np.std(valid_rmse_correct)
    else:
        RMSE_correct_mean = RMSE_correct_std = float('inf')

    # # 观测边指标汇总
    # observed_edge_correct = sum(metrics[0] for metrics in observed_edge_metrics)
    # observed_edge_add = sum(metrics[1] for metrics in observed_edge_metrics)
    # observed_edge_miss = sum(metrics[2] for metrics in observed_edge_metrics)
    
    # # 计算观测边的总体指标
    # observed_edge_PRE, observed_edge_REC, observed_edge_F1 = _calculate_metrics(
    #     observed_edge_correct, observed_edge_add, observed_edge_miss
    # )
    # observed_edge_PRE, observed_edge_REC, observed_edge_F1 = _calculate_filtered_means(observed_edge_metrics)

    observed_edge_PRE = np.mean([observed_edge_metrics[i][0] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) if len([observed_edge_metrics[i][0] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0
    observed_edge_REC = np.mean([observed_edge_metrics[i][1] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) if len([observed_edge_metrics[i][1] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0
    observed_edge_F1 = np.mean([observed_edge_metrics[i][2] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) if len([observed_edge_metrics[i][2] for i in range(len(observed_edge_metrics)) if is_cluster_correct_list[i] == True]) !=0 else 0.0

    print(observed_edge_metrics)
    print(observed_edge_PRE, observed_edge_REC, observed_edge_F1)
    print(edge_metrics)
    print(edge_PRE, edge_REC, edge_F1)
    return [
        is_cluster_correct_count, is_latent_edge_correct_count, 
        is_observed_edge_correct_count, is_totally_correct_count,
        cluster_PRE, cluster_REC, cluster_F1,
        edge_PRE, edge_REC, edge_F1,
        observed_edge_PRE, observed_edge_REC, observed_edge_F1,
        RMSE_mean, RMSE_std,
        RMSE_correct_mean, RMSE_correct_std
    ]


def _calculate_filtered_means(metrics_list):
    """计算过滤掉零值后的平均值"""
    filtered_metrics = [[m for m in metric if m != 0] for metric in zip(*metrics_list)]
    
    means = []
    for filtered_metric in filtered_metrics:
        if filtered_metric:
            means.append(np.mean(filtered_metric))
        else:
            means.append(0.0)
    
    return means
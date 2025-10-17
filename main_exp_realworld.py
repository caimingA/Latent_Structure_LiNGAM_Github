import sys
sys.path.append("./LaHiCaSl")
import numpy as np
import pandas as pd
import utils
import lingam
import ReLVLiNGAM_once
import LSLiNGAM as LSL
import LSLiNGAM_claude as LSL_C
from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
import random

# from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
from scipy.stats import pearsonr
from causallearn_local.search.HiddenCausal.GIN.GIN import GIN
from causallearn_local.search.HiddenCausal.GIN.GIN import GIN_MI

from causallearn_local.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

import optimized_lslingam
import LSLiNGAM_memory as LSL_M
import LSLiNGAM_claude as LSL_M_C
import ReLVLiNGAM

import LaHiCaSl.LaHiCaSl as Xie

# semopyをインポートする
import semopy as sem
# 仮説モデルをインスタンス化するModelもインポートする
from semopy import Model
# semopyライブラリからサンプルデータをインポートする
from semopy.examples import political_democracy
from semopy.examples import holzinger39

import seaborn as sns


# # Visualization using pydot
# from causallearn.utils.GraphUtils import GraphUtils
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import io

def add_noise(X, noise_level=0.03, n_aug=3):
    X_aug = [X]
    for _ in range(n_aug):
        noise = np.random.normal(0, noise_level * X.std(axis=0), X.shape)
        X_aug.append(X + noise)
    return np.vstack(X_aug)

# if __name__ == '__main__':
#     # data = political_democracy.get_data()
#     data = holzinger39.get_data()
#     print(data.head())

#     # data_select = data[['x1', 'y1', 'y3', 'y8']]
#     # # 中心化
#     # data_select = (data_select - data_select.mean(axis=0))

#     data_select = data[['x1', 'x4', 'x7', 'x8']]
#     # 中心化
#     data_select = (data_select - data_select.mean(axis=0))
#     # / data_select.std(axis=0)

#     # # # 画出各个维度的直方图
#     # # data_select.hist(bins=30, figsize=(8, 6))
#     # # plt.tight_layout()
#     # # plt.show()

#     # print(data_select.head())

#     data_select_list = np.array(data_select.values.tolist())
#     # data_select_list = add_noise(data_select_list, n_aug=3)
#     print(data_select_list.shape)
#     model = LSL_C.LSLiNGAM(
#             data_select_list,
#             1,
#             ind_alpha=0.5,
#             one_latent_tol=0.01,
#             singular_threshold=0.01,
#             merge_threshold_first=0.01,
#             merge_threshold_next=0.05
#         )
#     model.fit()

#     print(model.latent_adjmatrix)
#     print(model.directed_edge_within_observed)

# political_democracy
if __name__ == '__main__':
    data = political_democracy.get_data()
    # data = holzinger39.get_data()
    print(data.head())

    # sns.pairplot(data)
    # plt.show()

    # ##################
    # # 要中心化！
    # data_select = data[['x1', 'y3', 'y5', 'y7']]
    
    # ##################
    # data_select = data
    data_select = data[['x1', 'x2', 'y3', 'y4', 'y5', 'y6']]

    # data_select = data[['x1', 'x2', 'y3', 'y4', 'y5', 'y6']]

    # data_select = data[['x1', 'x2', 'x3', 'y1', 'y4', 'y6', 'y7']]
    # data_select = data[['x1', 'y1', 'y3', 'y6', 'y8']]
    # data_select = data[['x1', 'y3', 'y5', 'y6']]
    # # 中心化
    # data_select = (data_select - data_select.mean(axis=0))

    # data_select = data[['x1', 'x4', 'x7', 'x8']]
    # 中心化
    # data_select = (data_select - data_select.mean(axis=0))
    data_select = (data_select - data_select.mean(axis=0)) / data_select.std(axis=0)

    # print(data_select.head())
    sns.pairplot(data_select)
    plt.show()

    # # # 画出各个维度的直方图
    # # data_select.hist(bins=30, figsize=(8, 6))
    # # plt.tight_layout()
    # # plt.show()

    # print(data_select.head())

    data_select_list = np.array(data_select.values.tolist())
    # data_select_list = add_noise(data_select_list, n_aug=3)
    print(data_select_list.shape)
    # model = LSL_C.LSLiNGAM(
    #         data_select_list,
    #         1,
    #         ind_alpha=0.5,
    #         one_latent_tol=0.05,
    #         singular_threshold=0.005,
    #         merge_threshold_first=0.01,
    #         merge_threshold_next=0.05
    #     )
    ###############################
    # model = LSL_C.LSLiNGAM(
    #         data_select_list,
    #         2,
    #         ind_alpha=0.2,
    #         one_latent_tol=0.1,
    #         singular_threshold=0.005,
    #         merge_threshold_first=0.001,
    #         merge_threshold_next=0.001
    #     )
    # model.fit()
###############################################################
    model = LSL_C.LSLiNGAM(
            data_select_list,
            2,
            ind_alpha=0.2,
            one_latent_tol=0.1,
            singular_threshold=0.005,
            merge_threshold_first=0.001,
            merge_threshold_next=0.01
        )
    model.fit()

    print(model.latent_adjmatrix)
    print(model.directed_edge_within_observed)

    print("=" * 20)
    G, K, latent_adjmatrix = GIN(data_select_list, alpha=0.2)
    print("GIN result:")
    print(K)
    print(latent_adjmatrix)
    # res_GIN_list.append(utils.evaluate_one(latent_adjmatrix, K, dict(), example_B_1[:hidden_num, :hidden_num], cluster_true, observed_edge_true, is_proposed=False))

    print("=" * 20)
    G, K, latent_adjmatrix = GIN_MI(data_select_list)
    print("GIN_MI result:")
    print(K)
    print(latent_adjmatrix)

    print("=" * 20)
    X_pd = pd.DataFrame(data_select_list) 
    print(X_pd.shape)

    res = Xie.Latent_Hierarchical_Causal_Structure_Learning(X_pd, 0.2)

    print("LaHiCaSl result:", res)
    # res_GIN_MI_list.append(utils.evaluate_one(latent_adjmatrix, K, dict(), example_B_1[:hidden_num, :hidden_num], cluster_true, observed_edge_true, is_proposed=False))
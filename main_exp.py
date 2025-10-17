import numpy as np
import utils
import lingam
import ReLVLiNGAM_once
import LSLiNGAM as LSL
from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat

from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
from scipy.stats import pearsonr
from causallearn.search.HiddenCausal.GIN.GIN import GIN


if __name__ == '__main__':
    # np.random.seed(2025)
    # np.random.seed(0)
    # case 1
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0],
    #         [1, 0, 0, 0],
    #         [1, 0, 0, 0],
    #         [1, 0, 1, 0]
    #     ],
    #     dtype= float
    # )
    # hidden_num = 1
    # cluster_true = [[0, 1, 2]]
    
    # # case 2
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #     ],
    #     dtype= float
    # )
    # hidden_num = 2
    # cluster_true = [[0], [1, 2]]

    # # case 3
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #         [0, 1, 0, 1, 0],
    #     ],
    #     dtype= float
    # )
    # hidden_num = 2
    # cluster_true = [[0], [1, 2]]
    
    # # case 4
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 1, 0],
    #     ],
    #     dtype= float
    # )
    # hidden_num = 2
    # cluster_true = [[0], [1, 2, 3]]

    # # case 5
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0], # L_1
    #         [1, 0, 0, 0, 0, 0, 0], # L_2
    #         [0, 1, 0, 0, 0, 0, 0], # L_3
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0],
    #     ],
    #     dtype= float
    # )
    # hidden_num = 3
    # cluster_true = [[0], [1], [2, 3]]

    # case 6
    example_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0], # L_1
            [1, 0, 0, 0, 0, 0, 0], # L_2
            [0, 1, 0, 0, 0, 0, 0], # L_3
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
        ],
        dtype= float
    )
    hidden_num = 3
    cluster_true = [[0], [1], [2, 3]]

    # # case 7
    # example_matrix = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0], # L_1
    #         [1, 0, 0, 0, 0, 0, 0], # L_2
    #         [1, 1, 0, 0, 0, 0, 0], # L_3
    #         [1, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0],
    #     ],
    #     dtype= float
    # )
    # hidden_num = 3
    # cluster_true = [[0], [1], [2, 3]]

    exp_num = 50
    sample_size = 100000

    res_list = list()
    res_list_GIN = list()
    cluster_list = list()
    
    for exp in range(exp_num):
        node_num = len(example_matrix)
        # example_disturbances_1 = utils.generate_disturbances("gamma", node_num, sample_size, [])
        # example_disturbances_1 = utils.generate_disturbances("uniform", node_num, sample_size, [])
        # example_disturbances_1 = utils.generate_disturbances("lognormal", node_num, sample_size, [])
        example_disturbances_1 = utils.generate_disturbances("exponential", node_num, sample_size, [])
        example_B_1 = utils.generate_coefficient_matrix(example_matrix, hidden_num)
        
        X_1, X_1_hidden, X_1_observed, Mixing_1, Mixing_1_hidden, Mixing_1_observed = utils.generate_data(example_disturbances_1, example_B_1, hidden_num)
        # std_

        model = LSL.LSLiNGAM(X_1_observed, 1)
        model.fit()
        
        cluster_est = model.ordered_cluster
        G_est = model.latent_adjmatrix

        res_list.append(utils.evaluate_one(G_est, cluster_est, example_B_1[:hidden_num, :hidden_num], cluster_true, is_proposed=True))
        
        
        print(cluster_est)
        print(G_est)
        print(exp, "th experiment")
        # print(example_B_1[:hidden_num, :hidden_num])
        G, K = GIN(X_1_observed[:1000, :])

        # print(G.graph)
        # print(K)

        cluster_est_GIN = K
        # graph_est_GIN = G.graph
        for i in range(len(G.graph[0])):
            if sum([i for i in G.graph[i,:]]) == 0 and sum([i for i in G.graph[:, i]]) == 0:
                cluster_est_GIN.append([i])
        for c in cluster_est_GIN:
            sorted(c)
        cluster_est_GIN = sorted(cluster_est_GIN, key=lambda x: x[0])

        cluster_num = len(cluster_est_GIN)
        graph_est_GIN = np.zeros((cluster_num, cluster_num))
        graph_est_GIN = G.graph[:cluster_num, :cluster_num]


        res_list_GIN.append(utils.evaluate_one(graph_est_GIN, cluster_est_GIN, example_B_1[:hidden_num, :hidden_num], cluster_true, is_proposed=False))

        # print(cluster_est_GIN)
        # print(graph_est_GIN)
    
        # print(model.latent_adjmatrix)
        # print(model.ordered_cluster)
    print(res_list)
    print(res_list_GIN)
    print(utils.evaluate_repeat(res_list))
    print(utils.evaluate_repeat(res_list_GIN))


        


        
    
import sys
# sys.path.append("./LaHiCaSl")
import numpy as np
import pandas as pd
import MakeGraph
import itertools
#Phase I: Locate latent variables
import IdentifyGlobalCausalClusters  # Stage I-S1
import Determine_Latent_Variables # Stage I-S2
import UpdateActiveData # Stage I-S3
#Phase II: Infer causal structure among latent variables
import LocallyInferCausalStructure #Stage II

debug = 1


def Latent_Hierarchical_Causal_Structure_Learning(data, alpha):

    """

    Latent Hierarchical Causal Structure Learning (LaHiCaSL)

    Parameters:
    data : set of observed variables
    alpha: Threshold

    Returns:
    Causal_Matrix : Causal structure matrix over both observed and latent variables

    """

    #Initialize latent index
    L_index=1
    #Initialize the ora data
    Ora_data = data.copy()
    # Initialize the graph
    LatentIndex = {}
    # Initialize CLuster set, that recored each learning result for different iteration
    AllPureCluster = []
    AllLearnedClusters = []
    AllImpureCluster = []
    # Initialize the signficant level
    # alpha = 0.05




    #Phase I: Locate latent variables
    print('Begin with Phase I: Locate latent variables +++++++++++++++++++')

    while(True):

        # Stage I-S1 ← IdentifyGlobalCausalClusters



        Cluster = IdentifyGlobalCausalClusters.IdentifyGlobalCausalClusters(data ,LatentIndex ,Ora_data, alpha)

        #if there not any new latent variable (cluster) is found
        if not Cluster or all(not v for v in Cluster.values()):
            break


        #Stage I-S2 ← DetermineLatentVariables

        Cluster, LatentIndex, data, PureCluster = Determine_Latent_Variables.MergerCluster(Cluster, data, Ora_data, LatentIndex, alpha)

        #All cluster is early learning, (that is a measurement-model)
        if not Cluster:
            break


        #Stage I-S3 ← UpdateActiveData

        LatentIndex, data, L_index = UpdateActiveData.UpdateActiveData(Cluster, L_index, LatentIndex, data, PureCluster, Ora_data)

        AllLearnedClusters, AllImpureCluster, AllPureCluster, LatentIndex = UpdateActiveData.UpdateAllClusterInformation(Cluster, PureCluster, AllLearnedClusters, AllPureCluster, AllImpureCluster, LatentIndex)

        if debug:
            print('=========> The Impure Cluster Set: ', AllImpureCluster)

    print('End of Phase I: Locate latent variables +++++++++++++++++++')


    #Phase II: Infer causal structure among latent variables


    print('Begin with Phase II: Infer causal structure among latent variables +++++++++++++++++++')

    LatentIndex = LocallyInferCausalStructure.LocalLearningStructure(AllImpureCluster, LatentIndex, Ora_data)

    print('End of Phase II: Infer causal structure among latent variables +++++++++++++++++++')



    Causal_Matrix = MakeGraph.UpdateGraph(list(Ora_data.columns),LatentIndex)

    print('================ The result of structure learning (adj matrix) : \n', Causal_Matrix)


    latent_num = len(LatentIndex)
    print("latent_num:", latent_num)
    adj_matrix = Causal_Matrix.values
    print("adj_matrix:", adj_matrix)
    observed_num = adj_matrix.shape[0] - latent_num
    adj_matrix_temp = np.zeros_like(adj_matrix, dtype=float)
    adj_matrix_temp[:latent_num, :latent_num] = adj_matrix[observed_num:, observed_num:]
    adj_matrix_temp[:latent_num, latent_num:] = adj_matrix[observed_num:, :observed_num]
    adj_matrix_temp[latent_num:, :latent_num] = adj_matrix[:observed_num, observed_num:]
    adj_matrix_temp[latent_num:, latent_num:] = adj_matrix[:observed_num, :observed_num]
    adj_matrix_temp = adj_matrix_temp.T
    
    adj_matrix_latent = adj_matrix_temp[:latent_num, :latent_num]
    observed_edge = {}
    for i in range(latent_num, latent_num + observed_num):
        for j in range(latent_num, latent_num + observed_num):
            if adj_matrix_temp[i][j] != 0: #j -> i
                if j not in observed_edge:
                    observed_edge[j] = []
                observed_edge[j].append(i)
    cluster = []
    cluster_set = set()
    for k, v in LatentIndex.items():
        cluster.append(v)
        for i in v:
            cluster_set.add(i)
        # cluster_set.add(tuple(v))
    print(cluster_set)
    print(set([i for i in range(observed_num)]))
    one_element_list = list(set([i for i in range(observed_num)]) - cluster_set)
    print("one_element_list:", one_element_list)
    temp_list = []
    for i in one_element_list:
        temp_list.append([i])
    cluster = temp_list + cluster

    adj_matrix_latent_extended = np.zeros((len(cluster), len(cluster)))
    if latent_num != 0:
        adj_matrix_latent_extended[len(temp_list): , len(temp_list): ] = adj_matrix_latent
    # print(Cluster)
    # print(LatentIndex)
    #Draw a graph
    #MakeGraph.Make_graph(LatentIndex)
    return adj_matrix_latent_extended, cluster, observed_edge
import numpy as np
import math
from numpy.linalg import svd
from scipy.stats import chi2
import copy
from collections import defaultdict, deque

from sympy import symbols, Number
from moment_estimation_c import estimate_moment
from constraints_to_test import get_constraints_for_l_latents, get_cumulant_formula, calculate_orders_needed
import itertools as it
from functools import reduce
from itertools import combinations, permutations
from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
import utils
import lingam
from scipy.stats import pearsonr

from typing import List, Iterator, Union

from causallearn_local.utils.KCI.KCI import KCI_UInd
from causallearn_local.graph.GeneralGraph import GeneralGraph
from causallearn_local.graph.GraphNode import GraphNode
from causallearn.graph.NodeType import NodeType
from causallearn_local.graph.Edge import Edge
from causallearn_local.graph.Endpoint import Endpoint

from hyppo.independence import Hsic

class LSLiNGAM():
    # def __init__(self, X, highest_l, verbose=False, only_lowest_order_equations=True, threshold_power=1/8, thresholds=[0.008, 0.8], scale_partly=True):
    def __init__(self, X, highest_l, ind_alpha, one_latent_tol, singular_threshold, merge_threshold_first, merge_threshold_next,scale_partly=True):
        # default
        # self._ind_alpha = 0.05
        # self._one_latent_tol = 0.02
        # self._singular_threshold = 0.01
        # self._merge_threshold_first = 0.01
        # self._merge_threshold_next = 0.25
        self._ind_alpha = ind_alpha
        self._one_latent_tol = one_latent_tol
        self._singular_threshold = singular_threshold
        self._merge_threshold_first = merge_threshold_first
        self._merge_threshold_next = merge_threshold_next
        
        self.X = X
        self.p = X.shape[1]
        self.n = X.shape[0] if X.shape[0] <= 2000 else 2000
        self.initial_indices = range(self.p)
        self.clusters = []
        self.ordered_cluster = [] # the clusters have been ordered
        self.adjmatrix = []
        self.latent_adjmatrix = []
        # self.D_record = dict()
        self.D_record = {}
        self.X_se = []
        self.X_se_now = []
        self.X_se_reverse = []

        self.highest_l = highest_l
        self.verbose = False
        self.only_lowest_order_equations = True
        self.scale_partly = scale_partly
        # self.threshold_power = threshold_power
        # self.thresholds = thresholds

        self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
        self.highest_order = calculate_orders_needed(highest_l)[1]

        # self.orders = range(2, self.highest_order+1)
        # self.omegas = np.full((self.p, len(self.orders)), np.nan)
        # self.pairwise_confounders = np.full((self.p, self.p), -1, dtype=int)
        self.upper_bounds_confounders = np.full((self.p, self.p), np.inf)
        # self.topological_order = []
        # self.B = np.full((self.p, self.p), np.nan)
        # self.descendants_latents = []
        # self.fitted = False
        self.generalGraph = GeneralGraph([])
        self.directed_edge_within_observed = dict()

    
    def fit(self):
        # pass
        self.cluster_Triad(self.X)
        # self.cluster_gin(self.X)
        print("clusters first: ", self.clusters)
        # self.clusters = [[0], [1], [2, 3]]
        # self.clusters = [[0], [1, 2]]
        # self.clusters = [[0], [1], [2], [3]]
        self.cluster_edge(self.X)
        print("clusters edge: ", self.clusters)
        self.cluster_merge()
        print("clusters: ", self.clusters)
        
        self.select_oldest()
        
        if len(self.clusters) == 1:
            self.ordered_cluster.append(self.clusters[0])
            self.latent_adjmatrix = np.array([[0.0]])
            self.make_result_graph()
            return self.ordered_cluster
        
        self.find_current_root()
        print(15*"=", "first end", "="*15)
        print("A_list: ", [{i: self.D_record[i]["A_list"]} for i in self.D_record.keys()])
        print("X_list: ", [{i: self.D_record[i]["X_list"]} for i in self.D_record.keys()])
        
        print("clusters after first: ", self.clusters)
        print("ordered clusters after first: ", self.ordered_cluster)

        while len(self.clusters) != 0:
            self.find_next_root()
            print(self.clusters)
            print("A_list: ", [{i: self.D_record[i]["A_list"]} for i in self.D_record.keys()])
            print("X_list: ", [{i: self.D_record[i]["X_list"]} for i in self.D_record.keys()])
        
        
        print(15*"=", "next end", "="*15)
        # print(self.X_se)
        # print(self.D_record)

        print("A_list: ", [{i: self.D_record[i]["A_list"]} for i in self.D_record.keys()])
        print("X_list: ", [{i: self.D_record[i]["X_list"]} for i in self.D_record.keys()])
        print("clusters after second: ", self.clusters)
        print("ordered clusters after second: ", self.ordered_cluster)

        print("X_se: ", self.X_se)
        cluster_num = len(self.ordered_cluster)

        self.X_se = []
        for c in self.ordered_cluster:
            for i in range(len(c)):
                if c[i] in self.D_record.keys():
                    self.X_se.append(c[i])

        self.X_se_reverse = self.X_se[::-1]

        if len(self.ordered_cluster) >= 2:
            self.remove_redundant_edge()
        self.make_result_graph()



    def make_result_graph(self):
        # G = GeneralGraph([])
        # for var in self.initial_indices:
        #     o_node = GraphNode(f"X{var + 1}")
        #     self.generalGraph.add_node(o_node)

        latent_id = 1
        l_nodes = list()
        count_end = 0
        for cluster in self.ordered_cluster:
            print(cluster)
            l_node = GraphNode(f"L{latent_id}")
            l_node.set_node_type(NodeType.LATENT)
            # print(l_node.get_node_type())
            self.generalGraph.add_node(l_node)
            count_start = 0
            for l in l_nodes:
                # print(count_start, " and ", count_end)
                if self.latent_adjmatrix[count_end][count_start] != 0.0:
                    self.generalGraph.add_directed_edge(l, l_node)
                count_start += 1
            l_nodes.append(l_node)

            for o in cluster:
                o_node = GraphNode(f"X{o + 1}")
                self.generalGraph.add_node(o_node)
                self.generalGraph.add_directed_edge(l_node, o_node)
            latent_id += 1
            count_end += 1


            for o in cluster:
                o_node = GraphNode(f"X{o + 1}")
                if o in self.directed_edge_within_observed.keys():
                    for o_1 in self.directed_edge_within_observed[o]:
                        o_1_node = GraphNode(f"X{o_1 + 1}")
                        self.generalGraph.add_directed_edge(o_node, o_1_node)
        # print(self.generalGraph.get_node(name="L1").get_node_type())
        # print(self.generalGraph.get_node(name="L2").get_node_type())

    
    
    # is used for clustering
    def find_all_confounders(self, X):
        """caculate the number of pair latent confounder in each pair of observed variables """
        remaining_nodes = range(X.shape[1]) 
        
        cumulants = self._estimate_cumulants(X) # all cumulant
        
        all_singular_values = self._calculate_all_singular_values(remaining_nodes, cumulants) # all singular values
        print("singular values: ", all_singular_values)

        # all confounders of each pair
        confounders = np.array([[self._estimate_num_confounders(potential_source, other_node, all_singular_values) if potential_source != other_node else 0 for potential_source in remaining_nodes] for other_node in remaining_nodes])

        # # print("confounders within: ", confounders)
        # for i in range(len(confounders)):
        #     for j in range(0, i):
        #         if i != j:
        #             if confounders[i][j] == 1 and confounders[j][i] == 1:
        #                 not_have_edge = test_one_confounder_sixth_robust(X[:, [i, j]], self._one_latent_tol)
        #                 if not_have_edge:
        #                     pass
        #                 else:
        #                     r = self.constraints[1]["r"]
        #                     sigma = all_singular_values[f"{i}{j}{1}"]
        #                     sigma_rev = all_singular_values[f"{j}{i}{1}"]
        #                     if sigma[r] / sigma[0] < sigma_rev[r] / sigma_rev[0]:
        #                         if i in self.directed_edge_within_observed.keys():
        #                             self.directed_edge_within_observed[i].append(j)
        #                         else:
        #                             self.directed_edge_within_observed[i] = [j]
        #                     else:
        #                         if j in self.directed_edge_within_observed.keys():
        #                             self.directed_edge_within_observed[j].append(i)
        #                         else:
        #                             self.directed_edge_within_observed[j] = [i]

        #                 # l = confounders[i][j]
        #                 # if l == np.inf:
        #                 #     l = self.highest_l
        #                 # r = self.constraints[l]["r"]
        #                 # sigma = all_singular_values[f"{i}{j}{l}"]
        #                 # sigma_rev = all_singular_values[f"{j}{i}{l}"]
        #                 # sigma_max = max(sigma[r], sigma_rev[r])

        #                 # print(np.fabs(sigma[r] - sigma_rev[r]))
        #                 # print(np.fabs(sigma[r] - sigma_rev[r]) / sigma_max)
        #                 # if np.fabs(sigma[r] - sigma_rev[r])> 0.01:
        #                 #     if sigma[r] < sigma_rev[r]:
        #                 #         if l == np.inf:
        #                 #             confounders[j][i] = self.highest_l
        #                 #         confounders[i][j] = confounders[j][i] + 1
        #                 #     else:
        #                 #         if l == np.inf:
        #                 #             confounders[i][j] = self.highest_l
        #                 #         confounders[j][i] = confounders[j][i] + 1
                                    
        return confounders
    
    
    def Triad_ind(self, i, j, k): 
        """
        Input: indices 
        """
        data = self.X[:, [i, j, k]]
        # data_centered = data
        data_centered = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
        print("test ind")
        kci = KCI_UInd()
        cov_matrix_ik = np.cov(data_centered[:, 0], data_centered[:, 2], bias=False)
        cov_matrix_jk = np.cov(data_centered[:, 1], data_centered[:, 2], bias=False)
        e_triad = data_centered[:, 0] - (cov_matrix_ik[1][0] / cov_matrix_jk[1][0])*data_centered[:, 1]
        # p_value, _ = kci.compute_pvalue(e_triad[:self.n, None], data_centered[:self.n, [2]])
        # print(self.X.shape)
        
        # cov_matrix_ik = np.cov(self.X[:, i], self.X[:, k], bias=False)
        # cov_matrix_jk = np.cov(self.X[:, j], self.X[:, k], bias=False)
        # e_triad = self.X[:, i] - (cov_matrix_ik[1][0] / cov_matrix_jk[1][0])*self.X[:, j]
        
        # hsic
        # _, p_value = hsic_test_gamma(e_triad[:self.n, None], self.X[:self.n, [k]])
        # print(e_triad[:1000].shape, " and ", self.X[:1000, k].shape)
        
        # p_value, _ = kci.compute_pvalue(e_triad[:self.n, None], data_centered[:self.n, [2]])

        _, p_value = Hsic(compute_kernel="laplacian").test(e_triad[:self.n, None], data_centered[:self.n, [2]])
        print("p_value: ", p_value)
        is_independent = p_value > self._ind_alpha

        return is_independent


    def cluster_Triad(self, X):
        remaining_nodes = self.initial_indices
        possible_num = len(remaining_nodes) - 2
        for index_pair in combinations(remaining_nodes, 2):
            # print(index_pair)
            i = index_pair[0]
            j = index_pair[1]
            count = 0
            for k in remaining_nodes:
                print(index_pair, " and ", k)
                
                # print(count)
                if k!= i and k != j:
                    count += 1
                    is_independent = self.Triad_ind(i, j, k)
                    if not is_independent:
                        break
                    else:
                        if count == possible_num:
                            self.clusters.append([i, j])
                            print([i, j], " is a cluster")
                            break
        cluster_set = set()
        for i in self.clusters:
            cluster_set.update(i)
        one_element_set = (set(remaining_nodes) - cluster_set)
        for e in one_element_set:
            self.clusters.append([e])

        # # add edge
        # for c in self.clusters:
        #     if len(c) != 1:
        #         i = c[0]
        #         j = c[1]
        #         confounders = self.find_all_confounders(X[:, c])
        #         print("confounders: ", confounders)
        #         if confounders[0][1] < confounders[1][0]:
        #                 if j in self.directed_edge_within_observed.keys():
        #                     self.directed_edge_within_observed[j].append(i)
        #                 else:
        #                     self.directed_edge_within_observed[j] = [i]
        #         if confounders[0][1] > confounders[1][0]:
        #             if i in self.directed_edge_within_observed.keys():
        #                 self.directed_edge_within_observed[i].append(j)
        #             else:
        #                 self.directed_edge_within_observed[i] = [j]
        # print("directed edges: ", self.directed_edge_within_observed)

        for c in self.clusters:
            if len(c) != 1:
                not_have_edge = test_one_confounder_sixth_robust(X[:, c], self._one_latent_tol)
                print("now cluster: ", c)
                if not_have_edge:
                    i = c[0]
                    j = c[1]
                    confounders = self.find_all_confounders(X[:, c])
                    print("confounders: ", confounders)
                    if confounders[0][1] < confounders[1][0]:
                            if j in self.directed_edge_within_observed.keys():
                                self.directed_edge_within_observed[j].append(i)
                            else:
                                self.directed_edge_within_observed[j] = [i]
                    if confounders[0][1] > confounders[1][0]:
                        if i in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                else:
                    i = c[0]
                    j = c[1]
                    remaining_nodes = range(X[:, c].shape[1])
                    cumulants = self._estimate_cumulants(X[:, c])
                    all_singular_values = self._calculate_all_singular_values(remaining_nodes, cumulants)
                    l = 1
                    r = self.constraints[l]["r"]
                    sigma = all_singular_values[f"{0}{1}{l}"]
                    rank_i_to_j = sigma[r] / sigma[0]
                    sigma = all_singular_values[f"{1}{0}{l}"]
                    rank_j_to_i = sigma[r] / sigma[0]
                    if rank_i_to_j < rank_j_to_i:
                        if i in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                    else:
                        if j in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[j].append(i)
                        else:
                            self.directed_edge_within_observed[j] = [i] 
        print("directed edges: ", self.directed_edge_within_observed)
    

    def cal_e_with_gin(self, data, cov, X, Z):
        cov_m = cov[np.ix_(Z, X)]
        _, _, v = np.linalg.svd(cov_m)
        omega = v.T[:, -1]
        return np.dot(data[:, X], omega)
    
    def cluster_gin(self, X):
        kci = KCI_UInd()
        cov = np.cov(X.T)
        print("cov shape: ", cov)
        remaining_nodes = self.initial_indices
        # possible_num = len(remaining_nodes) - 2
        var_set = set(range(len(remaining_nodes)))
        for cluster in combinations(var_set, 2):
            print("gin clusters: ", cluster)
            remain_var_set = list(var_set - set(cluster))
            print("gin remain set: ", remain_var_set)
            e = self.cal_e_with_gin(X, cov, list(cluster), list(remain_var_set))
            pvals = []
            for z in range(len(remain_var_set)):
                pvals.append(kci.compute_pvalue(X[:self.n, [remain_var_set[z]]], e[:self.n, None])[0])
            print(pvals)
            fisher_pval = fisher_test(pvals)
            print("fisher_pval", fisher_pval)
            if fisher_pval >= self._ind_alpha:
                self.clusters.append(list(cluster))
                print(list(cluster), " is a cluster")
        cluster_set = set()
        for i in self.clusters:
            cluster_set.update(i)
        one_element_set = (set(remaining_nodes) - cluster_set)
        for e in one_element_set:
            self.clusters.append([e])
        
        for c in self.clusters:
            if len(c) != 1:
                not_have_edge = test_one_confounder_sixth_robust(X[:, c], self._one_latent_tol)
                if not_have_edge:
                    i = c[0]
                    j = c[1]
                    confounders = self.find_all_confounders(X[:, c])
                    print("confounders: ", confounders)
                    if confounders[0][1] < confounders[1][0]:
                            if j in self.directed_edge_within_observed.keys():
                                self.directed_edge_within_observed[j].append(i)
                            else:
                                self.directed_edge_within_observed[j] = [i]
                    if confounders[0][1] > confounders[1][0]:
                        if i in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                else:
                    i = c[0]
                    j = c[1]
                    remaining_nodes = range(X[:, c].shape[1])
                    cumulants = self._estimate_cumulants(X[:, c])
                    all_singular_values = self._calculate_all_singular_values(remaining_nodes, cumulants)
                    l = 1
                    r = self.constraints[l]["r"]
                    sigma = all_singular_values[f"{0}{1}{l}"]
                    rank_i_to_j = sigma[r] / sigma[0]
                    sigma = all_singular_values[f"{1}{0}{l}"]
                    rank_j_to_i = sigma[r] / sigma[0]
                    if rank_i_to_j < rank_j_to_i:
                        if i in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                    else:
                        if j in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[j].append(i)
                        else:
                            self.directed_edge_within_observed[j] = [i] 
        print("directed edges: ", self.directed_edge_within_observed)


    def cluster_edge(self, X):
        # remaining_nodes = self.initial_indices
        clusters = self.clusters
        nodes_possible = list()
        nodes_possible_update = set()
        for c in clusters:
            if len(c) == 1:
                nodes_possible.append(c[0])

        # nodes_num = len(nodes_possible)
        # remaining_nodes = 
        for index_pair in combinations(nodes_possible, 2):
            # print(index_pair)
            i = index_pair[0]
            j = index_pair[1]
            print([i, j])
            not_have_edge = test_one_confounder_sixth_robust(X[:, index_pair], self._one_latent_tol)
            
            # ######
            # if [i, j] == [1, 3] or [i, j] == [2, 3]:
            #     not_have_edge = False
            # else:
            #     not_have_edge = True
            # ######
            # print(not_have_edge)
            
            # print("confounders pair: ", self.find_all_confounders(X[:, [i, j]]))
            if not_have_edge:
                continue
            else:
                nodes_possible_update.add(i)
                nodes_possible_update.add(j)
        nodes_possible_update = list(nodes_possible_update)
        nodes_num = len(nodes_possible_update)
        print(nodes_possible_update)
        confounders = self.find_all_confounders(X[:, nodes_possible_update])
        # #######
        # confounders = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # #######
        
        print("confounders: ", confounders)
        print("confounders: ", self.directed_edge_within_observed)
        for i in range(nodes_num):
            for j in range(nodes_num):
                if confounders[i][j] != confounders[j][i]:
                    if i < j:
                        self.clusters.append([nodes_possible_update[i], nodes_possible_update[j]])
                    if confounders[i][j] > confounders[j][i]:
                        if nodes_possible_update[i] in self.directed_edge_within_observed.keys():
                            self.directed_edge_within_observed[nodes_possible_update[i]].append(nodes_possible_update[j])
                        else:
                            self.directed_edge_within_observed[nodes_possible_update[i]] = [nodes_possible_update[j]]
        # for i in range(nodes_num):
        #     for j in range(nodes_num):
        #         if confounders[i][j] != confounders[j][i]:
        #             if i < j:
        #                 self.clusters.append([nodes_possible_update[i], nodes_possible_update[j]])
        #             if confounders[i][j] > confounders[j][i]:
        #                 if nodes_possible_update[j] in self.directed_edge_within_observed.keys():
        #                     self.directed_edge_within_observed[nodes_possible_update[j]].append(nodes_possible_update[i])
        #                 else:
        #                     self.directed_edge_within_observed[nodes_possible_update[j]] = [nodes_possible_update[i]]
                    # else:
                    #     if i in self.directed_edge_within_observed.keys():
                    #         self.directed_edge_within_observed[nodes_possible_update[i]].append(nodes_possible_update[j])
                    #     else:
                    #         self.directed_edge_within_observed[nodes_possible_update[i]] = [nodes_possible_update[j]]


    def cluster_merge(self):
        clusters = copy.deepcopy(self.clusters)
        print(clusters)
        # clusters_num = len(clusters)
        parent = {}
        
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        
        for lst in clusters:
            if not lst:
                continue
            first = lst[0]
            parent.setdefault(first, first)

            for x in lst[1:]:
                parent.setdefault(x, x)
                union(first, x)

        groups = defaultdict(set)
        for x in parent:
            groups[find(x)].add(x)

        # 输出结果：每组转换回有序列表
        clusters = [sorted(g) for g in groups.values()]
        
        clusters = sorted(clusters, key=lambda x: x[0])
        
        self.clusters = clusters
        print("within: ", clusters) 
        
         
    def _estimate_cumulants(self, X):
        """Estimate all cumulants that are relevant for ReLVLiNGAM, i.e. all cumulants with up to two distinct indices."""
        # For efficiency reasons, first estimate all moments and then plug them into the cumulant formulas instead of estimating each cumulant separately.
        nodes = range(X.shape[1])
        nodes_num = len(nodes)
        moment_dict = self._estimate_moments(X)
        all_cumulants = {}
        # print("cumulant")
        for k in range(2, self.highest_order+1):
            # print("c: ", k)
            kth_cumulant = np.array([get_cumulant_formula(ind).subs(moment_dict) if len(set(ind)) <= 2 else np.nan for ind in it.product(range(nodes_num), repeat = k)], dtype=float).reshape((nodes_num,)*k)
            all_cumulants.update({k: kth_cumulant})
        return all_cumulants
    

    def _estimate_moments(self, X):
        """Estimate all moments that are relevant for ReLVLiNGAM, i.e. all moments with up to two distinct indices."""
        # nodes = range(self.p)
        nodes = range(X.shape[1])
        nodes = sorted(nodes)
        moment_dict = {}
        # print("moment")
        for k in range(2, self.highest_order+1):
            # print("m: ", k)
            moment_dict.update({symbols(f"m_{''.join(map(str, ind))}"): estimate_moment(np.array(ind), X) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return moment_dict
    
    
    def _form_symbol_to_cumulant_dict(self, cumulants, nodes, scale_partly):
        nodes = sorted(nodes)
        nodes_num = len(nodes)
        highest_k = len(cumulants) + 1
        cumulant_dict = {}
        # print(cumulants)
        # scaling
        if scale_partly:
            scales = np.array([cumulants[2][i,i]**(1/2) if i in nodes else np.nan for i in range(nodes_num)])
            for k in range(2, highest_k+1):
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind]/np.prod(scales[list(ind)]) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        else:
            # print("k: ", k, "->", cumulants[3][1, 2])
            for k in range(2, highest_k+1):
                # print("k: ", k, "->", cumulants[k][0, 1])
                # print("ind: ", [ind for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2])
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind] for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return cumulant_dict
    

    def _calculate_all_singular_values(self, remaining_nodes, cumulants):
        """Calculate all singular values for all pairs of remaining nodes."""
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, remaining_nodes, self.scale_partly)
        # cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, remaining_nodes, scale_partly=False)
        sigmas = {}
        for (potential_source, other_node) in it.combinations(remaining_nodes, 2):
            for l in range(self.highest_l+1):
                r = self.constraints[l]["r"]
                A, A_rev = self.constraints[l]["A"], self.constraints[l]["A_rev"]
                specify_nodes = {sym: symbols("c_" + "".join(sorted(sym.name[2:].replace("j", str(potential_source)).replace("i", str(other_node))))) for sym in A.free_symbols | A_rev.free_symbols}
                A_hat = np.array(A.subs(specify_nodes).subs(cumulant_dict), dtype=float)
                A_rev_hat = np.array(A_rev.subs(specify_nodes).subs(cumulant_dict), dtype=float)
                sigma = svd(A_hat, compute_uv=False)
                sigma_rev = svd(A_rev_hat, compute_uv=False)
                sigmas[f"{potential_source}{other_node}{l}"] = sigma.tolist()
                sigmas[f"{other_node}{potential_source}{l}"] = sigma_rev.tolist()
                # print((potential_source, other_node))
                # print(potential_source,"->", other_node, ": ", np.linalg.matrix_rank(A_hat))
                # print(other_node,"->", potential_source, ": ", np.linalg.matrix_rank(A_rev_hat))
        return sigmas
    
    
    def _estimate_num_confounders(self, potential_source, other_node, all_singular_values):
        """Estimate the number of confounders between two nodes."""
        print("source: ", potential_source, "tail: ", other_node)
        # iteration = len(self.topological_order)
        # threshold = self.thresholds[0]/self.n**self.threshold_power if iteration == 0 else self.thresholds[1]*iteration/self.n**self.threshold_power
        threshold = self._singular_threshold
        # threshold_power = 1/8
        # threshold = 0.008/self.n**threshold_power
        highest_l = min(self.upper_bounds_confounders[other_node, potential_source], self.highest_l)
        for l in range(highest_l+1):
            r = self.constraints[l]["r"]
            sigma = all_singular_values[f"{potential_source}{other_node}{l}"]
            
            # print("r: ", r)
            print("l: ", l, "sigma: ", sigma)
            print(sigma[r]/sigma[0])
            # print(sigma[r]/sigma[r-1])
            # for s in range(len(sigma)):
            # print("cum: ", [s/np.sum(sigma) for s in sigma])
            # print("threshold: ", threshold)
            if (sigma[r]/sigma[0] < threshold):
                print("l: ", l)
                return l
            # if (sigma[r]/np.sum(sigma) < threshold):
            #     print("l: ", l)
            #     return l
            # if (sigma[r]/sigma[0] < threshold):
            #     return l
            # 
            # if (sigma[r]/sigma[r-1] < threshold):
            #     return l
            # 
            # if (sigma[r] < threshold):
            #     return l
        return self.upper_bounds_confounders[other_node, potential_source]

    
    def estimate_latent_cumulants(self, j, i, cumulants, not_edge=False):
        """
        j -> i, j的顺位比i高
        """
        l = self.highest_l
        # highest_l = 1+l
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, [j, i], scale_partly=False)
        # print("cumulant_dict: ", cumulant_dict)
        equations_bij = self.constraints[l]["equations_bij"]

        specify_nodes = {sym: symbols(sym.name[:2] + "".join(sorted(sym.name[2:].replace("j", str(j)).replace("i", str(i))))) for sym in reduce(set.union, [eq.free_symbols for eq in equations_bij]) if str(sym) != "b_ij"}
        
        all_roots = np.full((l+1,len(equations_bij)), np.nan)
        for e in range(len(equations_bij)):
            eq = equations_bij[e]
            # print("eq: ", eq)
            # Need type conversion for numpy root function to work
            # estimated_coeffs = [coeff.subs(specify_nodes).subs(cumulant_dict) for coeff in eq.all_coeffs()]
            
            # print(eq.all_coeffs())
            estimated_coeffs = [float(coeff.subs(specify_nodes).subs(cumulant_dict)) for coeff in eq.all_coeffs()]
            print("estimated_coeffs: ", estimated_coeffs)
            # A numpy polynomial has the opposite order of coefficients to sympy: Numpy starts with the lowest power, 
            # Sympy with the highest. Therefore, reverse the coefficients.
            roots = np.polynomial.Polynomial(estimated_coeffs[::-1]).roots()
        
            if len(roots) < l+1:
                print(f"Warning: {l} confounders were estimated but corresponding equation does only have {len(roots)} roots. Roots are {roots}.")
                missing = l+1 - len(roots)
                roots = np.append(roots, [np.nan]*missing)
            # roots = np.sort(np.real(roots))
            roots = np.real(roots)
            all_roots[:,e] = roots # 列向量
        print("all root: ", all_roots.T)
        mean_roots = np.nanmean(all_roots, axis=1)
        print("root: ", mean_roots)
        if not_edge == True:
            mean_roots = np.array(sorted(mean_roots, key=abs))
        
        print("root: ", mean_roots)
        k_2 = calculate_orders_needed(l)[1] - 1
        B_tilde = [mean_roots**i for i in range(k_2)]
        print("B_tilde: ", B_tilde)
        # print(f"c_{''.join(sorted((str(j),)*(k_2 - index) + (str(i),)*index))}" for index in range(k_2))
        y = np.array([float(cumulant_dict[symbols(f"c_{''.join(sorted((str(j),)*(k_2 - index) + (str(i),)*index))}")]) for index in range(k_2)])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        # return all_roots
        return marginal_omegas
    

    def estimate_latent_cumulants_1(self, data, not_edge=True):
        """
        data是一对变量的观测数据
        """
        # coefficients = [
        #                 cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 1]) - cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 1, 1])
        #                 , cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 1, 1]) - cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 0])
        #                 , cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 0, 0]) - cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 0, 1])
        #                 ]
        coefficients = [
                        cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 1, 1, 1]) - cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 1, 1])
                        , cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 0, 1]) - cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 1, 1, 1])
                        , cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 0, 1, 1]) - cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 0, 0, 1])
                        ]
        # coefficients = [-np.linalg.det(A_3), np.linalg.det(A_2), -np.linalg.det(A_1), np.linalg.det(A_0)][::-1]
        # if not_edge:
        #     coefficients[0] = 0.0
        roots = np.polynomial.Polynomial(coefficients).roots()
        print("root: ", roots)
        mean_roots = np.array(sorted(roots, key=abs))
        print("root_mean: ", mean_roots)
        if not_edge:
            mean_roots[0] = 0.0
        B_tilde = [mean_roots**i for i in range(3)]
        print("B_tilde", B_tilde)
        # y = np.array([cumulant(data, [0, 0, 0]), cumulant(data, [0, 0, 1]), cumulant(data, [0, 1, 1])])
        y = np.array([cumulant(data, [0, 0, 0]), cumulant(data, [0, 0, 1]), cumulant(data, [0, 1, 1])])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        # return all_roots
        return marginal_omegas


    def estimate_latent_cumulants_2(self, data, not_edge=True):
        """
        data是一对变量的观测数据
        """
        # A_3 = [
        #     [cumulant(data, [0, 0, 0, 0]), cumulant(data, [0, 0, 0, 1]), cumulant(data, [0, 0, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1])],
        # ]

        # A_2 = [
        #     [cumulant(data, [0, 0, 0, 0]), cumulant(data, [0, 0, 0, 1]), cumulant(data, [0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 1, 1, 1, 1])],
        # ]

        # A_1 = [
        #     [cumulant(data, [0, 0, 0, 0]), cumulant(data, [0, 0, 1, 1]), cumulant(data, [0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1])],
        # ]

        # A_0 = [
        #     [cumulant(data, [0, 0, 0, 1]), cumulant(data, [0, 0, 1, 1]), cumulant(data, [0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1])],
        # ]
        # A_3 = [
        #     [cumulant(data, [0, 0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1])],
        # ]

        # A_2 = [
        #     [cumulant(data, [0, 0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1, 1])],
        # ]

        # A_1 = [
        #     [cumulant(data, [0, 0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1, 1])],
        # ]

        # A_0 = [
        #     [cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1])],
        #     [cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1, 1])],
        # ]
        A = np.array(
            [
                [cumulant(data, [0, 0, 0, 0]), cumulant(data, [0, 0, 0, 1]), cumulant(data, [0, 0, 1, 1]), cumulant(data, [0, 1, 1, 1])],
                [cumulant(data, [0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1])],
                [cumulant(data, [0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 1, 1]), cumulant(data, [0, 0, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1])],
                [cumulant(data, [0, 0, 0, 0, 0, 0]), cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1])],
                [cumulant(data, [0, 0, 0, 0, 0, 1]), cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1])],
                [cumulant(data, [0, 0, 0, 0, 1, 1]), cumulant(data, [0, 0, 0, 1, 1, 1]), cumulant(data, [0, 0, 1, 1, 1, 1]), cumulant(data, [0, 1, 1, 1, 1, 1])],
            ]
        )
        # coefficients = [-np.linalg.det(A_3), np.linalg.det(A_2), -np.linalg.det(A_1), np.linalg.det(A_0)][::-1]
        # coefficients[0] = 0.0
        res_root = list()
        max_zero = 1000
        for rows in combinations([0, 1, 2, 3, 4, 5], 3):
            rows = sorted(rows)
            print(rows)
            A_3 = A[rows, :][:, [0, 1, 2]]
            print(A_3)
            A_2 = A[rows, :][:, [0, 1, 3]]
            A_1 = A[rows, :][:, [0, 2, 3]]
            A_0 = A[rows, :][:, [1, 2, 3]]
            coefficients = [-np.linalg.det(A_3), np.linalg.det(A_2), -np.linalg.det(A_1), np.linalg.det(A_0)][::-1]
            # print(coefficients)
            # coefficients[0] = 0.0
            # print(coefficients)
            roots = np.polynomial.Polynomial(coefficients).roots()
            roots = np.real(roots)
            # print("root: ", roots)
            mean_roots = np.array(sorted(roots, key=abs))
            # print("sorted_roots: ", mean_roots)
            if mean_roots[0] < max_zero:
                res_root = mean_roots
        print(res_root)
        # if not_edge:
        #     coefficients[0] = 0.0
        # roots = np.polynomial.Polynomial(coefficients).roots()
        # print("root: ", roots)
        # mean_roots = np.array(sorted(roots, key=abs))
        # B_tilde = [mean_roots**i for i in range(3)]
        if not_edge:
            res_root[0] = 0.0
        B_tilde = [res_root**i for i in range(3)]
        print("B_tilde", B_tilde)
        y = np.array([cumulant(data, [0, 0, 0]), cumulant(data, [0, 0, 1]), cumulant(data, [0, 1, 1])])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        # return all_roots
        return marginal_omegas
    

    def select_oldest(self):
        clusters = self.clusters
        for c in clusters:
            self.X_se.append(c[0])
            self.X_se_now.append(c[0])
            self.D_record[c[0]] = {"A_list": [], "X_list": [], "S_list": [], "P_list": []}
        print(self.X_se)
    
    
    def find_current_root(self):
        print(15*"=", "first", "="*15)
        data = self.X
        X_se = copy.deepcopy(self.X_se)
        
        clusters = copy.deepcopy(self.clusters)
        cluster_num = len(clusters)

        true_mapping_dict = dict()

        for i in range(len(X_se)):
            true_mapping_dict[X_se[i]] = i # ture index： virtual index
        
        cluster_within_map_dict = dict() # another index of variable which belongs to the same cluster as the variable in X_se
        for i in range(cluster_num):
            if len(clusters[i]) == 1:
                cluster_within_map_dict[X_se[i]] = -1
            else:
                cluster_within_map_dict[X_se[i]] = clusters[i][1]
        print(cluster_within_map_dict)
        
        true_res_index = -1
        cluster_res_index = -1

        if cluster_num == 1:
            self.ordered_cluster.append(self.clusters[0])
            self.X_se_now.pop(0) # res_index -> possible_root -> the index in X_se
            self.clusters.pop(0)
        else:
            self.highest_order = calculate_orders_needed(1)[1]
            roots_not_possible = set()
            if cluster_num > 2:
                for index_pair in combinations(X_se, 2):
                    i = index_pair[0]
                    j = index_pair[1]
                    print([i, j])
                    not_have_edge = test_one_confounder_sixth_robust(data[:, index_pair], self._one_latent_tol)
                    print(not_have_edge)
                    if not_have_edge: # condition 1
                        continue
                    else:
                        roots_not_possible.add(i)
                        roots_not_possible.add(j)

            print("root_not_possible: ", roots_not_possible)
            self.highest_l = 1
            roots_possible = list(set(X_se) - roots_not_possible)
            print("root_possible: ", roots_possible)
            if len(roots_possible) == 0:
                roots_possible = list(roots_not_possible)
                print("root_possible: ", roots_possible)
            if len(roots_possible) == 1 and cluster_num != 2:
                # res_index = true_mapping_dict[roots_possible[0]]
                res_index = 0
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
            else:
                if cluster_num == 2:
                    roots_possible = X_se
                print("root_possible: ", roots_possible)
                pair_cumulant_matrix = np.zeros((len(roots_possible), len(roots_possible)))
                pair_cumulant_list = list()
                mapping_dict = dict()
                
                for i in roots_possible:
                    j = cluster_within_map_dict[i]
                    if j == -1:
                        # pair_cumulant_list.append(-1)
                        pair_cumulant_list.append([])
                    else:    
                        print("within: ", [i, j])
                        pair_cumulant_within = self._estimate_cumulants(data[:, [i, j]])
                        # pair_cumulant_list.append(self.estimate_latent_cumulants(0, 1, pair_cumulant_within)[1])
                        pair_cumulant_list.append(self.estimate_latent_cumulants(0, 1, pair_cumulant_within))

                
                for i in range(len(roots_possible)):
                    mapping_dict[roots_possible[i]] = i
                
                print("mapping_dict: ", mapping_dict)
                print("pair_cumulant_matrix: ", pair_cumulant_matrix)
                print("between")
                cumulants = self._estimate_cumulants(data[:, roots_possible])
                for index_pair in combinations(roots_possible, 2):                
                    i = index_pair[0]
                    j = index_pair[1]

                    print([i, j])
                    # mapping_dict[i] -> mapping_dict[j]
                    # pair_cumulant_matrix[mapping_dict[j]][mapping_dict[i]] = self.estimate_latent_cumulants(mapping_dict[i], mapping_dict[j], cumulants, not_edge=True)[1]
                    pair_data = data[:, [i, j]]
                    pair_cumulant_matrix[mapping_dict[j]][mapping_dict[i]] = self.estimate_latent_cumulants_1(pair_data)[1]
                    print([j, i])
                    # mapping_dict[j] -> mapping_dict[i]
                    # pair_cumulant_matrix[mapping_dict[i]][mapping_dict[j]] = self.estimate_latent_cumulants(mapping_dict[j], mapping_dict[i], cumulants, not_edge=True)[1]
                    pair_data = data[:, [j, i]]
                    pair_cumulant_matrix[mapping_dict[i]][mapping_dict[j]] = self.estimate_latent_cumulants_1(pair_data)[1]

                pair_cumulant_matrix = np.array(pair_cumulant_matrix)
                print("pair_cumulant_matrix: \n", pair_cumulant_matrix)
                result_list = list()
                result_list_mean = list()
                for i in range(len(roots_possible)):
                    # mean = np.mean(pair_cumulant_matrix[:, i])
                    if len(pair_cumulant_list[i]) == 0:
                        # result_list.append(np.var(pair_cumulant_matrix[:, i]))
                        nonzero_list = list()
                        for j in range(len(pair_cumulant_matrix)):
                            if i != j:
                                nonzero_list.append(pair_cumulant_matrix[j][i])
                        result_list.append(np.var(nonzero_list))                       
                        result_list_mean.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list[i][0])
                        nonzero_list = list()
                        for j in range(len(temp_list)):
                            if i != j:
                                nonzero_list.append(temp_list[j])
                        res_temp = np.var(nonzero_list)
                        for j in pair_cumulant_list[i]:
                            temp_temp_list = np.append(pair_cumulant_matrix[:, i], j)
                            print(temp_list)
                            nonzero_list = list()
                            for k in range(len(temp_temp_list)):
                                if i != k:
                                    nonzero_list.append(temp_temp_list[k])
                            res_temp_temp = np.var(nonzero_list)
                            if res_temp > res_temp_temp:
                                res_temp = res_temp_temp
                        result_list.append(res_temp)
                        result_list_mean.append(res_temp)
                
                print("res: ", result_list)
                print("res_mean: ", result_list_mean)
                
                res_index = np.argmin(result_list)

                print(res_index)
                print(true_mapping_dict[roots_possible[res_index]])

                # Merge
                # res_index 是 possible_root中的index，possible_root[res_index]是真实的节点index
                # i = 0
                # true_res_index = roots_possible[res_index]
                # while i != len(self.clusters):
                #     cluster_res_index = true_mapping_dict[true_res_index]
                #     if i != cluster_res_index:
                #         if X_se[i] in roots_possible:
                true_res_index = roots_possible[res_index] # 根节点的真实index
                cluster_res_index = true_mapping_dict[true_res_index] # 真实根所对应的cluster的index
                merge_cluster_list = list()
                for i in range(len(result_list)):
                    if i != res_index:
                        if np.abs(result_list[i] - result_list[res_index]) <=self._merge_threshold_first:
                            merge_cluster_list.append(true_mapping_dict[roots_possible[i]])
                merge_cluster_list.sort(reverse=True) # 真实cluster的index的倒序
                for i in merge_cluster_list:
                    self.clusters[cluster_res_index] += self.clusters[i]
                    self.clusters.pop(i) # pop(index)
                    self.D_record.pop(self.X_se_now[i]) # pop(key)
                    self.X_se_now.pop(i) # pop(index)
                    # self.D_record.pop(X_se[i]) # pop(key)
                    # X_se.pop(i) # pop(index)
                    # self.X_se.pop(i) # pop(index)
                    if i < cluster_res_index:
                        cluster_res_index -= 1

                # i = 0            
                # while i != len(self.clusters):
                #     if i != res_index:
                #         # print(res_index, "and ", i)
                #         # print(self.clusters)
                #         if np.abs(result_list[i] - result_list[res_index]) <=self._merge_threshold_first:
                #             # print(self.clusters[i], "and ", self.clusters[res_index])
                #             self.clusters[res_index] += self.clusters[i]
                #             self.clusters.pop(i) # pop(index)
                #             self.D_record.pop(X_se[i]) # pop(key)
                #             X_se.pop(i) # pop(index)
                #             self.X_se_now.pop(i) # pop(index)
                #             i -= 1 # pop后，i会变小
                #             if i < res_index:
                #                 res_index -= 1
                #     i += 1

                print("res_index: ", res_index)
                print("cluster_res_index: ", cluster_res_index)
            self.ordered_cluster.append(sorted(self.clusters[cluster_res_index]))
            self.X_se_now.pop(cluster_res_index) # 
            self.clusters.pop(cluster_res_index)
            # for x in X_se:
            for x in self.X_se_now:
                # if x!= roots_possible[res_index]:
                if x!= true_res_index:
                    self.D_record[x]["A_list"].append(0)
                    self.D_record[x]["X_list"].append(true_res_index)
                    self.D_record[x]["S_list"].append(data[:, true_res_index])
                    self.D_record[x]["P_list"].append(compute_new_e(data[:, x], data[:, true_res_index])[0])
        # self.X_se = X_se
        # self.X_se_now = X_se
            
            
            # self.ordered_cluster.append(sorted(self.clusters[true_mapping_dict[roots_possible[res_index]]]))
            # self.X_se_now.pop(true_mapping_dict[roots_possible[res_index]]) # res_index -> possible_root -> the index in X_se
            # self.clusters.pop(true_mapping_dict[roots_possible[res_index]])
            
            # self.ordered_cluster.append(sorted(self.clusters[true_mapping_dict[roots_possible[res_index]]]))
            # self.X_se_now.pop(res_index) # res_index -> possible_root -> the index in X_se
            # self.clusters.pop(res_index)
            # for x in X_se:
            #     if x!= roots_possible[res_index]:
            #         self.D_record[x]["A_list"].append(0)
            #         self.D_record[x]["X_list"].append(roots_possible[res_index])
            #         self.D_record[x]["S_list"].append(data[:, roots_possible[res_index]])
            #         self.D_record[x]["P_list"].append(compute_new_e(data[:, x], data[:, roots_possible[res_index]])[0])

        
    
    def find_next_root(self):
        print(15*"=", "next", "="*15)
        data = np.copy(self.X)
        X_se = copy.deepcopy(self.X_se_now)
        # X_se = copy.deepcopy(self.X_se)
        clusters = copy.deepcopy(self.clusters)
        cluster_num = len(clusters)
        print("clusters: ", clusters)

        true_mapping_dict = dict()

        for i in range(len(X_se)):
            true_mapping_dict[X_se[i]] = i # 真实index -> 顺位index
        
        cluster_within_map_dict = dict() # another index of variable which belongs to the same cluster as the variable in X_se
        for i in range(cluster_num):
            if len(clusters[i]) == 1:
                cluster_within_map_dict[X_se[i]] = -1
            else:
                cluster_within_map_dict[X_se[i]] = clusters[i][1]
        print(cluster_within_map_dict)

        true_res_index = -1
        cluster_res_index = -1
        
        if cluster_num == 1:
            # self.D_record[X_se[0]] = {"A_list": [], "X_list": [], "S_list": []}
            self.ordered_cluster.append(self.clusters[0])
            self.X_se_now.pop(0) # res_index -> possible_root -> the index in X_se
            self.clusters.pop(0)
        else:
            self.highest_order = calculate_orders_needed(1)[1]
            e_tilde_dict = {}
            roots_not_possible = set()
            if cluster_num >= 2:
                for index_pair in permutations(X_se, 2):
                    i = index_pair[0]
                    j = index_pair[1]
                    print([i, "->", j])
                    e_tilde = self.remove_all_influence(i) # 去掉现在的所有根节点的影响
                    if i in e_tilde_dict:
                        pass
                    else:
                        e_tilde_dict[i] = e_tilde
                    original_x = data[:, j]
                    pair_data = np.array([e_tilde, original_x]).T
                    # print(original_x)
                    print("pair data shape: ", pair_data.shape)
                    not_have_edge = test_one_confounder_sixth_robust(pair_data, self._one_latent_tol)
                    print(not_have_edge)
                    if not_have_edge: # condition 1
                        continue
                    else:
                        roots_not_possible.add(i)
            
            roots_possible = list(set(X_se) - roots_not_possible)
            
            print("possible roots: ", roots_possible)
            if len(roots_possible) == 0:
                # roots_possible = X_se
                roots_possible = list(roots_not_possible)
            if len(roots_possible) == 1 and cluster_num != 2:
                # res_index = true_mapping_dict[roots_possible[0]]
                res_index = 0
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
            else:  
                if cluster_num == 2:
                    roots_possible = X_se          
                print("possible roots: ", roots_possible)
                pair_cumulant_matrix = np.zeros((len(roots_possible), len(roots_possible)))
                pair_cumulant_list = list()
                pair_cumulant_list_2 = list()
                mapping_dict = dict()
                
                # 计算组内cumulant
                print("within")
                for i in roots_possible:
                    j = cluster_within_map_dict[i]
                    print([i, j])
                    if j == -1:
                        pair_cumulant_list.append(-1)
                        pair_cumulant_list_2.append(-1)
                    else:    
                        # print(e_tilde[i])
                        pair_data = np.array([e_tilde_dict[i], data[:, j]]).T
                        is_one_confounder = test_one_confounder_sixth_robust(pair_data, self._one_latent_tol)
                        if is_one_confounder:
                            print("one confounder")
                            self.highest_l = 1
                            self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
                            self.highest_order = calculate_orders_needed(1)[1]
                            # pair_cumulant_within = self._estimate_cumulants(pair_data)
                            cumulants_within = self.estimate_latent_cumulants_1(pair_data)
                            pair_cumulant_list.append(cumulants_within[1])
                            pair_cumulant_list_2.append(cumulants_within[1])
                        else:
                            print("two confounders")
                            self.highest_l = 2
                            self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
                            self.highest_order = calculate_orders_needed(2)[1]
                            cumulants_within = self.estimate_latent_cumulants_2(pair_data)
                            pair_cumulant_list.append(cumulants_within[1])
                            pair_cumulant_list_2.append(cumulants_within[2])
                            if i in self.directed_edge_within_observed.keys():
                                self.directed_edge_within_observed[i].append(j)
                            else:
                                self.directed_edge_within_observed[i] = [j]


                # 真实index -> 临时index
                for i in range(len(roots_possible)):
                    mapping_dict[roots_possible[i]] = i
                
                print(mapping_dict)
                print(pair_cumulant_matrix)
                
                # 计算组间cumulant
                print("between")
                self.highest_l = 1
                self.highest_order = calculate_orders_needed(1)[1]
                cumulants = self._estimate_cumulants(data[:, roots_possible])
                for index_pair in permutations(roots_possible, 2):                
                    i = index_pair[0]
                    j = index_pair[1]
                    print([i, j])
                    pair_data = np.array([e_tilde_dict[i], data[:, j]]).T
                    # cumulants = self._estimate_cumulants(pair_data)
                    pair_cumulant_matrix[mapping_dict[j]][mapping_dict[i]] = self.estimate_latent_cumulants_1(pair_data)[1]

                
                pair_cumulant_matrix = np.array(pair_cumulant_matrix)
                print("pair_cumulant_matrix: ", pair_cumulant_matrix)
                
                result_list = list()
                result_list_2 = list()
                result_mean_list = list()
                result_mean_list_2 = list()
                for i in range(len(roots_possible)):
                    if pair_cumulant_list[i] == -1:
                        nonzero_list = list()
                        for j in range(len(pair_cumulant_matrix)):
                            if i != j:
                                nonzero_list.append(pair_cumulant_matrix[j][i])
                        result_list.append(np.var(nonzero_list))
                        result_mean_list.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list[i])
                        print(temp_list)
                        nonzero_list = list()
                        for j in range(len(temp_list)):
                            if i != j:
                                nonzero_list.append(temp_list[j])
                        result_list.append(np.var(nonzero_list))
                        result_mean_list.append(np.mean(nonzero_list))                
                    if pair_cumulant_list_2[i] == -1:
                        nonzero_list = list()
                        for j in range(len(pair_cumulant_matrix)):
                            if i != j:
                                nonzero_list.append(pair_cumulant_matrix[j][i])
                        result_list_2.append(np.var(nonzero_list))
                        result_mean_list_2.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list_2[i])
                        print(temp_list)
                        nonzero_list = list()
                        for j in range(len(temp_list)):
                            if i != j:
                                nonzero_list.append(temp_list[j])
                        result_list_2.append(np.var(nonzero_list))
                        result_mean_list_2.append(np.mean(nonzero_list))

                print("res: ", result_list)
                print("res_2: ", result_list_2)
                print("res_mean: ", result_mean_list)
                print("res_mean_2: ", result_mean_list_2)
                
                for i in range(len(result_list)):
                    if result_list[i] > result_list_2[i]:
                        result_list[i] = result_list_2[i]
                        result_mean_list[i] = result_mean_list_2[i]
                
                print("res: ", result_list)
                print("res_mean: ", result_mean_list)
                
                res_index = np.argmin(result_list)
                
                print("root now: ", roots_possible[res_index])

                # ##### merge #####
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
                merge_cluster_list = list()
                for i in range(len(result_list)):
                    if i != res_index:
                        if np.abs(result_list[i] - result_list[res_index]) <=self._merge_threshold_next:
                            merge_cluster_list.append(true_mapping_dict[roots_possible[i]])
                merge_cluster_list.sort(reverse=True)
                for i in merge_cluster_list:
                    self.clusters[cluster_res_index] += self.clusters[i]
                    self.clusters.pop(i) # pop(index)
                    self.D_record.pop(self.X_se_now[i]) # pop(key)
                    self.X_se_now.pop(i) # pop(index)
                    # self.clusters.pop(i) # pop(index)
                    # self.D_record.pop(X_se[i]) # pop(key)
                    # X_se.pop(i) # pop(index)
                    # self.X_se.pop(i) # pop(index)
                    if i < cluster_res_index:
                        cluster_res_index -= 1
                # i = 0
                # while i != len(self.clusters):
                #     if i != res_index:
                #         if np.abs(result_list[i] - result_list[res_index]) <=self._merge_threshold_next:
                #             self.clusters[res_index] += self.clusters[i]
                #             self.clusters.pop(i) # pop(index)
                #             self.D_record.pop(X_se[i]) # pop(key)
                #             X_se.pop(i)
                #             self.X_se_now.pop(i)
                #             i -= 1 # pop后，i会变小
                #             if i < res_index:
                #                 res_index -= 1
                #     i += 1
                print("res_index: ", res_index)
                print("cluster_res_index: ", cluster_res_index)
            self.ordered_cluster.append(sorted(self.clusters[cluster_res_index]))
            self.X_se_now.pop(cluster_res_index) # 
            self.clusters.pop(cluster_res_index)
            
            now_cluster_index = len(self.ordered_cluster)-1
            e_tilde = self.remove_all_influence(true_res_index)
            
            # for x in X_se:
            for x in self.X_se_now:
                # if x!= roots_possible[res_index]:
                if x!= true_res_index:
                    self.D_record[x]["A_list"].append(now_cluster_index)
                    self.D_record[x]["X_list"].append(true_res_index)
                    pho = compute_new_e(data[:, x], e_tilde)[0]
                    self.D_record[x]["S_list"].append(e_tilde)
                    self.D_record[x]["P_list"].append(pho)


            # self.ordered_cluster.append(sorted(self.clusters[true_mapping_dict[roots_possible[res_index]]]))
            # self.X_se_now.pop(true_mapping_dict[roots_possible[res_index]]) # res_index -> possible_root -> the index in X_se
            # self.clusters.pop(true_mapping_dict[roots_possible[res_index]])

            # self.ordered_cluster.append(sorted(self.clusters[res_index]))
            # self.X_se_now.pop(res_index) # res_index -> possible_root -> the index in X_se
            # self.clusters.pop(res_index)

            # print("root now: ", roots_possible[res_index])
            # now_cluster_index = len(self.ordered_cluster)-1
            # print("X_se", X_se)
            # print("now_cluster_index", now_cluster_index)
            # e_tilde = self.remove_all_influence(roots_possible[res_index])
            # for x in X_se:
            #     print("x: ", x)
            #     print("roots_possible[res_index]: ", roots_possible[res_index])
            #     if x!= roots_possible[res_index]:
            #         print("x!= roots_possible[res_index]")
            #         self.D_record[x]["A_list"].append(now_cluster_index)
            #         self.D_record[x]["X_list"].append(roots_possible[res_index])
            #         pho = compute_new_e(data[:, x], e_tilde)[0]
            #         self.D_record[x]["S_list"].append(e_tilde)
            #         self.D_record[x]["P_list"].append(pho)
        # self.X_se = X_se
        # self.X_se_now = X_se
                    
    
    def remove_redundant_edge(self):
        print(15*"=", "remove", "="*15)
        X_se_reverse = self.X_se_reverse # 一定是一条路
        count = 0
        cluster_num = len(self.ordered_cluster)
        self.latent_adjmatrix = np.zeros((cluster_num, cluster_num), dtype=float)
        if cluster_num == 2:
            index_now = X_se_reverse[0]
            X_now = copy.deepcopy(self.X[:, index_now])
            previous_ancestor = self.D_record[index_now]["S_list"][0]
            pho = compute_new_e(X_now, previous_ancestor)[0]
            self.latent_adjmatrix[1][0] = pho
            return self.latent_adjmatrix
        
        while cluster_num - count >= 2:
            print("X_se_reverse: ", X_se_reverse)
            index_now = X_se_reverse[count]
            print("index now: ", index_now)
            X_now = copy.deepcopy(self.X[:, index_now])
            ancester_index_list = self.D_record[index_now]["X_list"][::-1]
            ancester_list = self.D_record[index_now]["S_list"][::-1]
            ancester_list_num = len(ancester_list)
            # index_previous_ancestor = 0
            # if len(ance)
            for index_ancestor in range(ancester_list_num):
                # index_previous_ancestor = 0
                X_index = ancester_index_list[index_ancestor]
                print("ancestor index now: ", X_index)
                previous_ancestor = ancester_list[index_ancestor]

                print("[", index_now, " -> ", X_index, "]")
                is_zero = is_coefficient_zero(X_now, previous_ancestor, self._ind_alpha)
                if is_zero:
                    self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = 0.0
                else:
                    pho_1 = compute_new_e(X_now, previous_ancestor)[0]
                    print("pho: ", pho_1)
                    if np.fabs(pho_1) < 0.5:
                        pho_1 = 0.0
                    self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = pho_1
                    X_now -= pho_1 * self.X[:, X_index]

                # index_previous_ancestor += 1
            count += 1


    def remove_all_influence(self, j):
        data = self.X
        X_j = data[:, j]

        S_list = self.D_record[j]["S_list"]
        P_list = self.D_record[j]["P_list"]

        e_tilde_j_s = copy.deepcopy(X_j)
        # print("S_list: ", S_list)
        # print("P_list: ", P_list)
        for index in range(len(P_list)):
            e_tilde_j_s -= S_list[index]*P_list[index]

        return e_tilde_j_s
    

def set_partitions(elements: List) -> Iterator[List[List]]:
    """
    递归生成集合的所有划分，每个划分表示为块（列表）的列表。
    
    Args:
        elements: 要划分的元素列表
        
    Yields:
        每个划分，表示为块的列表
        
    Examples:
        >>> list(set_partitions([0, 1]))
        [[[0, 1]], [[0], [1]]]
    """
    if not elements:
        yield []
        return
        
    if len(elements) == 1:
        yield [elements]
        return
    
    first = elements[0]
    rest = elements[1:]
    
    # 对剩余元素递归求划分
    for partition in set_partitions(rest):
        # 将first加入现有的每个块
        for i in range(len(partition)):
            new_partition = []
            for j, block in enumerate(partition):
                if i == j:
                    new_partition.append([first] + block)
                else:
                    new_partition.append(block[:])  # 创建副本避免修改原数据
            yield new_partition
        
        # first单独成块
        yield [[first]] + [block[:] for block in partition]


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
    # 遍历所有的划分
    for partition in set_partitions(indices_list):
        # 对每个划分，计算每个块的样本矩：E[∏_{j in block} X_j]
        prod_moment = 1.0
        for block in partition:
            # axis=1 表示对每个样本计算乘积，再对所有样本求均值
            vals = np.prod(data[:, block], axis=1)
            prod_moment *= vals.mean()
        # 权重为 (|partition|-1)! * (-1)^(|partition|-1)
        weight = math.factorial(len(partition) - 1) * ((-1) ** (len(partition) - 1))
        cumulant_value += weight * prod_moment
    return cumulant_value
    
    
def test_one_confounder_sixth_robust(data, one_latent_tol):
    """
    更robust的版本，处理边界情况
    """
    
    # 索引定义
    # indices_cum_4_2 = [0, 0, 0, 0, 1, 1]  # cum(X₁⁴X₂²)
    # indices_cum_2_4 = [0, 0, 1, 1, 1, 1]  # cum(X₁²X₂⁴)
    # indices_cum_3_3 = [0, 0, 0, 1, 1, 1]  # cum(X₁³X₂³)

    indices_cum_4_2 = [0, 0, 0, 1]  # cum(X₁⁴X₂²)
    indices_cum_2_4 = [0, 1, 1, 1]  # cum(X₁²X₂⁴)
    indices_cum_3_3 = [0, 0, 1, 1]  # cum(X₁³X₂³)
    
    # 数据中心化
    # data_centered = data
    data_centered = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
    
    # 计算累积量
    cum_4_2 = cumulant(data_centered, indices_cum_4_2)
    cum_2_4 = cumulant(data_centered, indices_cum_2_4)
    cum_3_3 = cumulant(data_centered, indices_cum_3_3)
    
    # 计算等式两边
    left_side = cum_4_2 * cum_2_4
    right_side = cum_3_3**2
    
    # 更robust的相对误差计算
    abs_left = abs(left_side)
    abs_right = abs(right_side)
    max_abs = max(abs_left, abs_right)
    
    if max_abs < 1e-12:  # 两边都接近0
        rel_err = 0.0
    else:
        # 使用较大值作为分母
        rel_err = abs(left_side - right_side) / max_abs
        # rel_err = abs(left_side - right_side) / right_side
        # rel_err = abs(left_side - right_side) / left_side
        # rel_err = abs(left_side - right_side) /1000
        # rel_err = abs(left_side - right_side) /100
        # rel_err = abs(left_side - right_side)
    
    tol = one_latent_tol
    
    print(f"cum(X₁⁴X₂²) × cum(X₁²X₂⁴) = {left_side}")
    print(f"cum(X₁³X₂³)² = {right_side}")
    print(f"绝对误差: {left_side - right_side}")
    print(f"相对误差: {rel_err}")
    print(f"宽容度: {tol}")
    
    return rel_err < tol


def compute_new_e(X_i, X_j):
    """
    X_i - rho*X_j
    """
    data = np.array([X_i, X_j]).T
    # print(data.shape)

    cum_iij = cumulant(data, [0, 0, 1, 1])
    cum_ijj = cumulant(data, [0, 1, 1, 1])

    rho = cum_iij / cum_ijj
    new_e = X_i - rho*X_j

    return rho, new_e


def is_coefficient_zero(X_i, X_j, ind_alpha):
    
    # _, p = hsic_test_gamma(X_i[:1000], X_j[:1000])
    # N = len(X_i) if len(X_i) <= 5000 else 5000
    N = len(X_i) if len(X_i) <= 2000 else 2000
    # kci = KCI_UInd()
    # p, _ = kci.compute_pvalue(X_i[:N, None], X_j[:N, None])
    data_centered = (np.array([X_i, X_j]).T - np.mean(np.array([X_i, X_j]).T, axis=0)) / np.std(np.array([X_i, X_j]).T, axis=0)
    _, p_value = Hsic(compute_kernel="laplacian").test(data_centered[:N, [0]], data_centered[:N, [1]])
    print("hsic: ", p_value)
    if p_value >= (ind_alpha/1.0):
        return True
    else:
        return False 


def calculate_cross_cumulant(X1, X2, k1, k2):
        """
        计算k1个X1和k2个X2的交叉累积量
        
        参数:
        X1, X2: numpy arrays, 观测数据
        k1: int, X1在累积量中出现的次数
        k2: int, X2在累积量中出现的次数
        
        返回:
        float: 交叉累积量的估计值
        """
        n_samples = len(X1)
        
        # 步骤1: 数据预处理和标准化
        X1_centered = X1 - np.mean(X1)
        X2_centered = X2 - np.mean(X2)
        
        # 步骤2: 计算所有可能的分区
        def generate_partitions(k1, k2):
            """生成所有可能的分区"""
            total_k = k1 + k2
            partitions = []
            
            # 使用多项式展开的系数来生成分区
            for p in range(1, total_k + 1):
                # 生成p个块的所有可能分区
                curr_partitions = []
                
                # 对于每个可能的X1分配方式
                for i in range(p + 1):
                    if i <= k1 and (p - i) <= k2:
                        curr_partitions.append((i, p - i))
                
                if curr_partitions:
                    partitions.append((p, curr_partitions))
            
            return partitions
        
        # 步骤3: 计算k阶矩
        def calculate_moment(X1, X2, p1, p2):
            """计算混合矩"""
            if p1 == 0:
                return np.mean(X2_centered ** p2)
            elif p2 == 0:
                return np.mean(X1_centered ** p1)
            else:
                return np.mean((X1_centered ** p1) * (X2_centered ** p2))
        
        # 步骤4: 实现k阶累积量的计算
        def calculate_joint_cumulant():
            result = 0.0
            partitions = generate_partitions(k1, k2)
            
            # 对每个分区计算其贡献
            for p, curr_partitions in partitions:
                partition_sum = 0
                
                # 计算当前分区的所有可能组合
                for p1, p2 in curr_partitions:
                    partition_sum += calculate_moment(X1, X2, p1, p2)
                
                # 使用Möbius反演公式计算累积量
                coef = (-1) ** (p - 1) * math.factorial(p - 1)
                result += coef * partition_sum
            
            return result / n_samples
        
        # 步骤5: 使用log-sum-exp技巧提高数值稳定性
        try:
            result = calculate_joint_cumulant()
            
            # 处理潜在的数值不稳定性
            if np.abs(result) < 1e-10:
                return 0.0
            
            return result
            
        except (RuntimeWarning, OverflowError) as e:
            # 如果出现数值问题，尝试使用更稳定的计算方法
            log_result = np.log(np.abs(result))
            sign = np.sign(result)
            
            return sign * np.exp(log_result)


def fisher_test(pvals):
    pvals = [pval if pval >= 1e-5 else 1e-5 for pval in pvals]
    fisher_stat = -2.0 * np.sum(np.log(pvals))
    return 1 - chi2.cdf(fisher_stat, 2 * len(pvals))
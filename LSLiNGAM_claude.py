import numpy as np
import math
import pandas as pd
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

from hyppo.independence import Hsic, MGC

class LSLiNGAM():
    def __init__(self, X, highest_l, ind_alpha, one_latent_tol, singular_threshold, merge_threshold_first, merge_threshold_next, scale_partly=True):
        self._ind_alpha = ind_alpha
        self._one_latent_tol = one_latent_tol
        self._singular_threshold = singular_threshold
        self._merge_threshold_first = merge_threshold_first
        self._merge_threshold_next = merge_threshold_next
        
        self.X = X
        self.p = X.shape[1]
        self.n = min(X.shape[0], 2000)
        self.initial_indices = range(self.p)
        self.clusters = []
        self.ordered_cluster = []
        self.latent_adjmatrix = []
        self.D_record = {}
        self.X_se_now = []

        self.highest_l = highest_l
        self.scale_partly = scale_partly

        self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
        self.highest_order = calculate_orders_needed(highest_l)[1]

        self.upper_bounds_confounders = np.full((self.p, self.p), np.inf)
        self.generalGraph = GeneralGraph([])
        self.directed_edge_within_observed = dict()

    
    def fit(self):
        self.cluster_Triad(self.X)
        print("clusters first: ", self.clusters)
        
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
        print("A_list: ", [{i: self.D_record[i]["A_list"]} for i in self.D_record.keys()])
        print("X_list: ", [{i: self.D_record[i]["X_list"]} for i in self.D_record.keys()])
        print("clusters after second: ", self.clusters)
        print("ordered clusters after second: ", self.ordered_cluster)

        cluster_num = len(self.ordered_cluster)

        if len(self.ordered_cluster) == 1:
            self.latent_adjmatrix = np.array([[0.0]])
        
        # 重新构建X_se，删除冗余变量
        X_se = []
        for c in self.ordered_cluster:
            for i in range(len(c)):
                if c[i] in self.D_record.keys():
                    X_se.append(c[i])

        X_se_reverse = X_se[::-1]

        if len(self.ordered_cluster) >= 2:
            # self.remove_redundant_edge(X_se_reverse)
            
            self.remove_redundant_edge_bootstrap(X_se_reverse)

        for k, v in self.directed_edge_within_observed.items():
            v=list(set(v))
            self.directed_edge_within_observed[k] = v

        for k, v in list(self.directed_edge_within_observed.items()):
            for i in v:
                if i in self.directed_edge_within_observed.keys():
                    if k in self.directed_edge_within_observed[i]:
                        if i < k:
                            self.directed_edge_within_observed[k].remove(i)
                            if len(self.directed_edge_within_observed[k]) == 0:
                                del self.directed_edge_within_observed[k]
                    else:
                        for j in self.directed_edge_within_observed[i]:
                            v.append(j)
                            
        
        
        # for k, v in list(self.directed_edge_within_observed.items()):
        #     for i in v:
        #         if i in self.directed_edge_within_observed.keys():
        #             if k in self.directed_edge_within_observed[i]:
        #                 if i < k:
        #                     self.directed_edge_within_observed[k].remove(i)
        #                     if len(self.directed_edge_within_observed[k]) == 0:
        #                         del self.directed_edge_within_observed[k]
        self.make_result_graph()

    def make_result_graph(self):
        latent_id = 1
        l_nodes = list()
        count_end = 0
        for cluster in self.ordered_cluster:
            print(cluster)
            l_node = GraphNode(f"L{latent_id}")
            l_node.set_node_type(NodeType.LATENT)
            self.generalGraph.add_node(l_node)
            
            count_start = 0
            for l in l_nodes:
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
    
    def find_all_confounders(self, X):
        """计算每对观测变量之间的潜在混杂变量数量"""
        remaining_nodes = range(X.shape[1]) 
        
        cumulants = self._estimate_cumulants(X)
        all_singular_values = self._calculate_all_singular_values(remaining_nodes, cumulants)
        print("singular values: ", all_singular_values)

        confounders = np.array([[self._estimate_num_confounders(potential_source, other_node, all_singular_values) if potential_source != other_node else 0 for potential_source in remaining_nodes] for other_node in remaining_nodes])
                                    
        return confounders
    
    def Triad_ind(self, i, j, k): 
        """检验三元组独立性"""
        data_view = self.X[:, [i, j, k]]
        data_centered = (data_view - data_view.mean(axis=0, keepdims=True)) / data_view.std(axis=0, keepdims=True)    
        
        print("test ind")
        
        cov_matrix_ik = np.cov(data_centered[:, 0], data_centered[:, 2], bias=False)
        cov_matrix_jk = np.cov(data_centered[:, 1], data_centered[:, 2], bias=False)
        e_triad = data_centered[:, 0] - (cov_matrix_ik[1][0] / cov_matrix_jk[1][0])*data_centered[:, 1]
        
        
        # X_k_ranking = rank_transform_1d(data_centered[:self.n, [2]])
        # e_triad_ranking = rank_transform_1d(e_triad[:self.n, None])
        # _, p_value = Hsic(compute_kernel="rbf").test(e_triad_ranking, X_k_ranking)

        # _, p_value = Hsic(compute_kernel="laplacian").test(e_triad[:self.n, None], data_centered[:self.n, [2]])
        _, p_value = Hsic(compute_kernel="polynomial").test(e_triad[:self.n, None], data_centered[:self.n, [2]])
        # _, p_value = Hsic(compute_kernel="rbf").test(e_triad[:self.n, None], data_centered[:self.n, [2]])

        # #############
        # _, p_value, _ = MGC().test(e_triad[:self.n, None], data_centered[:self.n, [2]])
        print("p_value: ", p_value)
        
        return p_value > self._ind_alpha

    def cluster_Triad(self, X):
        remaining_nodes = self.initial_indices
        possible_num = len(remaining_nodes) - 2
        
        for index_pair in combinations(remaining_nodes, 2):
            i, j = index_pair
            count = 0
            
            for k in remaining_nodes:
                print(index_pair, " and ", k)
                
                if k != i and k != j:
                    count += 1
                    is_independent = self.Triad_ind(i, j, k)
                    if not is_independent:
                        break
                    else:
                        if count == possible_num:
                            self.clusters.append([i, j])
                            print([i, j], " is a cluster")
                            break
        
        # 添加单个节点的聚类
        cluster_set = set()
        for cluster in self.clusters:
            cluster_set.update(cluster)
        one_element_set = set(remaining_nodes) - cluster_set
        for e in one_element_set:
            self.clusters.append([e])

        # 处理聚类内部的边
        for c in self.clusters:
            if len(c) != 1:
                ### trash version ###
                # not_have_edge = self.test_one_confounder_schkoda(X[:, c])
                #####################
                not_have_edge = test_one_confounder_sixth_robust(X[:, c], self._one_latent_tol)
                print("now cluster: ", c)
                
                if not_have_edge:
                    i, j = c[0], c[1]
                    confounders = self.find_all_confounders(X[:, c])
                    print("confounders: ", confounders)
                    
                    if confounders[0][1] < confounders[1][0]:
                        if j in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[j].append(i)
                        else:
                            self.directed_edge_within_observed[j] = [i]
                    elif confounders[0][1] > confounders[1][0]:
                        if i in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                else:
                    i, j = c[0], c[1]
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
                        if i in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                    else:
                        if j in self.directed_edge_within_observed:
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
        for cluster in self.clusters:
            cluster_set.update(cluster)
        one_element_set = set(remaining_nodes) - cluster_set
        for e in one_element_set:
            self.clusters.append([e])
        
        # 处理聚类内部的边
        for c in self.clusters:
            if len(c) != 1:
                ### trash version ###
                # not_have_edge = self.test_one_confounder_schkoda(X[:, c])
                #####################
                not_have_edge = test_one_confounder_sixth_robust(X[:, c], self._one_latent_tol)

                
                if not_have_edge:
                    i, j = c[0], c[1]
                    confounders = self.find_all_confounders(X[:, c])
                    print("confounders: ", confounders)
                    
                    if confounders[0][1] < confounders[1][0]:
                        if j in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[j].append(i)
                        else:
                            self.directed_edge_within_observed[j] = [i]
                    elif confounders[0][1] > confounders[1][0]:
                        if i in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                else:
                    i, j = c[0], c[1]
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
                        if i in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[i].append(j)
                        else:
                            self.directed_edge_within_observed[i] = [j]
                    else:
                        if j in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[j].append(i)
                        else:
                            self.directed_edge_within_observed[j] = [i] 
        print("directed edges: ", self.directed_edge_within_observed)

    def cluster_edge(self, X):
        nodes_possible = []
        for c in self.clusters:
            if len(c) == 1:
                nodes_possible.append(c[0])

        nodes_possible_update = set()
        for index_pair in combinations(nodes_possible, 2):
            i, j = index_pair
            print([i, j])

            ### trash version ###
            # not_have_edge = self.test_one_confounder_schkoda(X[:, index_pair])
            #####################
            not_have_edge = test_one_confounder_sixth_robust(X[:, index_pair], self._one_latent_tol)
            
            if not not_have_edge:
                nodes_possible_update.add(i)
                nodes_possible_update.add(j)
        
        nodes_possible_update = list(nodes_possible_update)
        nodes_num = len(nodes_possible_update)
        print(nodes_possible_update)
        
        confounders = self.find_all_confounders(X[:, nodes_possible_update])
        print("confounders: ", confounders)
        print("confounders: ", self.directed_edge_within_observed)
        
        for i in range(nodes_num):
            for j in range(nodes_num):
                if confounders[i][j] != confounders[j][i]:
                    if i < j:
                        self.clusters.append([nodes_possible_update[i], nodes_possible_update[j]])
                    if confounders[i][j] > confounders[j][i]:
                        if nodes_possible_update[i] in self.directed_edge_within_observed:
                            self.directed_edge_within_observed[nodes_possible_update[i]].append(nodes_possible_update[j])
                        else:
                            self.directed_edge_within_observed[nodes_possible_update[i]] = [nodes_possible_update[j]]

    def cluster_merge(self):
        clusters = copy.deepcopy(self.clusters)
        print(clusters)
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

        clusters = [sorted(g) for g in groups.values()]
        clusters = sorted(clusters, key=lambda x: x[0])
        
        self.clusters = clusters
        print("within: ", clusters) 
         
    def _estimate_cumulants(self, X):
        """估计所有相关的累积量"""
        nodes = range(X.shape[1])
        nodes_num = len(nodes)
        moment_dict = self._estimate_moments(X)
        all_cumulants = {}
        
        for k in range(2, self.highest_order+1):
            kth_cumulant = np.array([get_cumulant_formula(ind).subs(moment_dict) if len(set(ind)) <= 2 else np.nan for ind in it.product(range(nodes_num), repeat = k)], dtype=float).reshape((nodes_num,)*k)
            all_cumulants.update({k: kth_cumulant})
        return all_cumulants
    
    def _estimate_moments(self, X):
        """估计所有相关的矩"""
        nodes = range(X.shape[1])
        nodes = sorted(nodes)
        moment_dict = {}
        
        for k in range(2, self.highest_order+1):
            moment_dict.update({symbols(f"m_{''.join(map(str, ind))}"): estimate_moment(np.array(ind), X) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return moment_dict
    
    def _form_symbol_to_cumulant_dict(self, cumulants, nodes, scale_partly):
        nodes = sorted(nodes)
        nodes_num = len(nodes)
        highest_k = len(cumulants) + 1
        cumulant_dict = {}
        
        if scale_partly:
            scales = np.array([cumulants[2][i,i]**(1/2) if i in nodes else np.nan for i in range(nodes_num)])
            for k in range(2, highest_k+1):
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind]/np.prod(scales[list(ind)]) for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        else:
            for k in range(2, highest_k+1):
                cumulant_dict.update({symbols(f"c_{''.join(map(str, ind))}"): cumulants[k][ind] for ind in it.combinations_with_replacement(nodes, k) if len(set(ind)) <= 2})
        return cumulant_dict
    
    def _calculate_all_singular_values(self, remaining_nodes, cumulants):
        """计算所有剩余节点对的奇异值"""
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, remaining_nodes, self.scale_partly)
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
        return sigmas
    
    def _estimate_num_confounders(self, potential_source, other_node, all_singular_values):
        """估计两个节点之间的混杂变量数量"""
        print("source: ", potential_source, "tail: ", other_node)
        threshold = self._singular_threshold
        highest_l = min(self.upper_bounds_confounders[other_node, potential_source], self.highest_l)
        
        for l in range(highest_l+1):
            r = self.constraints[l]["r"]
            sigma = all_singular_values[f"{potential_source}{other_node}{l}"]
            
            print("l: ", l, "sigma: ", sigma)
            print(sigma[r]/sigma[0])
            
            if (sigma[r]/sigma[0] < threshold):
                print("l: ", l)
                return l
                
        return self.upper_bounds_confounders[other_node, potential_source]
    
    def estimate_latent_cumulants(self, j, i, cumulants, not_edge=False):
        """估计潜在累积量"""
        l = self.highest_l
        cumulant_dict = self._form_symbol_to_cumulant_dict(cumulants, [j, i], scale_partly=False)
        equations_bij = self.constraints[l]["equations_bij"]

        specify_nodes = {sym: symbols(sym.name[:2] + "".join(sorted(sym.name[2:].replace("j", str(j)).replace("i", str(i))))) for sym in reduce(set.union, [eq.free_symbols for eq in equations_bij]) if str(sym) != "b_ij"}
        
        all_roots = np.full((l+1, len(equations_bij)), np.nan)
        for e in range(len(equations_bij)):
            eq = equations_bij[e]
            estimated_coeffs = [float(coeff.subs(specify_nodes).subs(cumulant_dict)) for coeff in eq.all_coeffs()]
            print("estimated_coeffs: ", estimated_coeffs)
            
            roots = np.polynomial.Polynomial(estimated_coeffs[::-1]).roots()
        
            if len(roots) < l+1:
                print(f"Warning: {l} confounders were estimated but corresponding equation does only have {len(roots)} roots. Roots are {roots}.")
                missing = l+1 - len(roots)
                roots = np.append(roots, [np.nan]*missing)
                
            roots = np.real(roots)
            all_roots[:,e] = roots
            
        print("all root: ", all_roots.T)
        mean_roots = np.nanmean(all_roots, axis=1)
        print("root: ", mean_roots)
        
        if not_edge:
            mean_roots = np.array(sorted(mean_roots, key=abs))
        
        print("root: ", mean_roots)
        k_2 = calculate_orders_needed(l)[1] - 1
        B_tilde = [mean_roots**i for i in range(k_2)]
        print("B_tilde: ", B_tilde)
        
        y = np.array([float(cumulant_dict[symbols(f"c_{''.join(sorted((str(j),)*(k_2 - index) + (str(i),)*index))}")]) for index in range(k_2)])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        
        return marginal_omegas
    
    def estimate_latent_cumulants_1(self, data, not_edge=True):
        """估计一个潜在变量的累积量"""
        coefficients = [
                        cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 1, 1, 1]) - cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 1, 1])
                        , cumulant(data, [0, 0, 1, 1])*cumulant(data, [0, 0, 0, 1]) - cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 1, 1, 1])
                        , cumulant(data, [0, 0, 0, 0])*cumulant(data, [0, 0, 1, 1]) - cumulant(data, [0, 0, 0, 1])*cumulant(data, [0, 0, 0, 1])
                        ]
        
        roots = np.polynomial.Polynomial(coefficients).roots().real
        print("root: ", roots)
        mean_roots = np.array(sorted(roots, key=abs))
        print("root_mean: ", mean_roots)
        
        if not_edge:
            mean_roots[0] = 0.0
            
        B_tilde = [mean_roots**i for i in range(3)]
        print("B_tilde", B_tilde)
        
        y = np.array([cumulant(data, [0, 0, 0]), cumulant(data, [0, 0, 1]), cumulant(data, [0, 1, 1])])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        
        return marginal_omegas

    def estimate_latent_cumulants_2(self, data, not_edge=True):
        """估计两个潜在变量的累积量"""
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
            
            roots = np.polynomial.Polynomial(coefficients).roots()
            roots = np.real(roots)
            mean_roots = np.array(sorted(roots, key=abs))
            
            if mean_roots[0] < max_zero:
                res_root = mean_roots
                
        print(res_root)
        
        if not_edge:
            res_root[0] = 0.0
            
        B_tilde = [res_root**i for i in range(3)]
        print("B_tilde", B_tilde)
        
        y = np.array([cumulant(data, [0, 0, 0]), cumulant(data, [0, 0, 1]), cumulant(data, [0, 1, 1])])
        print("y: ", y)
        marginal_omegas = np.linalg.lstsq(B_tilde, y, rcond=None)[0]
        print("marginal_omegas: ", marginal_omegas)
        
        return marginal_omegas

    def select_oldest(self):
        for c in self.clusters:
            self.X_se_now.append(c[0])
            self.D_record[c[0]] = {"A_list": [], "X_list": [], "S_list": [], "P_list": []}
        print(self.X_se_now)
    
    def find_current_root(self):
        print(15*"=", "first", "="*15)
        data = self.X
        X_se = copy.deepcopy(self.X_se_now)
        clusters = copy.deepcopy(self.clusters)
        cluster_num = len(clusters)

        true_mapping_dict = {X_se[i]: i for i in range(len(X_se))}
        
        cluster_within_map_dict = {}
        for i in range(cluster_num):
            if len(clusters[i]) == 1:
                cluster_within_map_dict[X_se[i]] = -1
            else:
                cluster_within_map_dict[X_se[i]] = clusters[i][1]
        print(cluster_within_map_dict)

        if cluster_num == 1:
            self.ordered_cluster.append(self.clusters[0])
            self.X_se_now.pop(0)
            self.clusters.pop(0)
        else:
            self.highest_order = calculate_orders_needed(1)[1]
            roots_not_possible = set()
            
            if cluster_num > 2:
                for index_pair in combinations(X_se, 2):
                    i, j = index_pair
                    print([i, j])
                    ### trash version ###
                    # not_have_edge = self.test_one_confounder_schkoda(data[:, index_pair])
                    #####################
                    not_have_edge = test_one_confounder_sixth_robust(data[:, index_pair], self._one_latent_tol)
                    print(not_have_edge)
                    if not not_have_edge:
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
                res_index = 0
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
            else:
                if cluster_num == 2:
                    roots_possible = X_se
                print("root_possible: ", roots_possible)
                
                pair_cumulant_matrix = np.zeros((len(roots_possible), len(roots_possible)))
                pair_cumulant_list = []
                mapping_dict = {roots_possible[i]: i for i in range(len(roots_possible))}
                
                for i in roots_possible:
                    j = cluster_within_map_dict[i]
                    if j == -1:
                        pair_cumulant_list.append([])
                    else:    
                        print("within: ", [i, j])
                        pair_cumulant_within = self._estimate_cumulants(data[:, [i, j]])
                        pair_cumulant_list.append(self.estimate_latent_cumulants(0, 1, pair_cumulant_within))
                
                print("mapping_dict: ", mapping_dict)
                print("pair_cumulant_matrix: ", pair_cumulant_matrix)
                print("between")
                
                for index_pair in combinations(roots_possible, 2):                
                    i, j = index_pair
                    print([i, j])
                    
                    pair_data = data[:, [i, j]]
                    pair_cumulant_matrix[mapping_dict[j]][mapping_dict[i]] = self.estimate_latent_cumulants_1(pair_data)[1]
                    print([j, i])
                    
                    pair_data = data[:, [j, i]]
                    pair_cumulant_matrix[mapping_dict[i]][mapping_dict[j]] = self.estimate_latent_cumulants_1(pair_data)[1]

                pair_cumulant_matrix = np.array(pair_cumulant_matrix)
                print("pair_cumulant_matrix: \n", pair_cumulant_matrix)
                
                result_list = []
                result_list_mean = []
                
                for i in range(len(roots_possible)):
                    if len(pair_cumulant_list[i]) == 0:
                        nonzero_list = [pair_cumulant_matrix[j][i] for j in range(len(pair_cumulant_matrix)) if i != j]
                        result_list.append(np.var(nonzero_list))                       
                        result_list_mean.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list[i][0])
                        nonzero_list = [temp_list[j] for j in range(len(temp_list)) if i != j]
                        res_temp = np.var(nonzero_list)
                        
                        for j in pair_cumulant_list[i]:
                            temp_temp_list = np.append(pair_cumulant_matrix[:, i], j)
                            print(temp_list)
                            nonzero_list = [temp_temp_list[k] for k in range(len(temp_temp_list)) if i != k]
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

                # 合并处理
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
                merge_cluster_list = []
                
                for i in range(len(result_list)):
                    if i != res_index:
                        if np.abs(result_list[i] - result_list[res_index]) <= self._merge_threshold_first:
                            merge_cluster_list.append(true_mapping_dict[roots_possible[i]])
                            
                merge_cluster_list.sort(reverse=True)
                for i in merge_cluster_list:
                    self.clusters[cluster_res_index] += self.clusters[i]
                    self.clusters.pop(i)
                    self.D_record.pop(self.X_se_now[i])
                    self.X_se_now.pop(i)
                    if i < cluster_res_index:
                        cluster_res_index -= 1

                print("res_index: ", res_index)
                print("cluster_res_index: ", cluster_res_index)
                
            self.ordered_cluster.append(sorted(self.clusters[cluster_res_index]))
            self.X_se_now.pop(cluster_res_index)
            self.clusters.pop(cluster_res_index)
            
            for x in self.X_se_now:
                if x != true_res_index:
                    self.D_record[x]["A_list"].append(0)
                    self.D_record[x]["X_list"].append(true_res_index)
                    self.D_record[x]["S_list"].append(data[:, true_res_index])
                    self.D_record[x]["P_list"].append(compute_new_e(data[:, x], data[:, true_res_index])[0])
    
    def find_next_root(self):
        print(15*"=", "next", "="*15)
        data = self.X  # 使用视图而非拷贝
        X_se = copy.deepcopy(self.X_se_now)
        clusters = copy.deepcopy(self.clusters)
        cluster_num = len(clusters)
        print("clusters: ", clusters)

        true_mapping_dict = {X_se[i]: i for i in range(len(X_se))}
        
        cluster_within_map_dict = {}
        for i in range(cluster_num):
            if len(clusters[i]) == 1:
                cluster_within_map_dict[X_se[i]] = -1
            else:
                cluster_within_map_dict[X_se[i]] = clusters[i][1]
        print(cluster_within_map_dict)
        
        if cluster_num == 1:
            self.ordered_cluster.append(self.clusters[0])
            self.X_se_now.pop(0)
            self.clusters.pop(0)
        else:
            self.highest_order = calculate_orders_needed(1)[1]
            e_tilde_dict = {}
            roots_not_possible = set()
            
            if cluster_num >= 2:
                for index_pair in permutations(X_se, 2):
                    i, j = index_pair
                    print([i, "->", j])
                    
                    if i not in e_tilde_dict:
                        e_tilde_dict[i] = self.remove_all_influence(i)
                    
                    e_tilde = e_tilde_dict[i]
                    original_x = data[:, j]
                    pair_data = np.array([e_tilde, original_x]).T
                    print("pair data shape: ", pair_data.shape)
                    ### trash version required by prof ###
                    # not_have_edge = self.test_one_confounder_schkoda(pair_data)
                    ######################################
                    not_have_edge = test_one_confounder_sixth_robust(pair_data, self._one_latent_tol)
                    print(not_have_edge)
                    
                    if not not_have_edge:
                        roots_not_possible.add(i)
            
            roots_possible = list(set(X_se) - roots_not_possible)
            print("possible roots: ", roots_possible)
            
            if len(roots_possible) == 0:
                roots_possible = list(roots_not_possible)
                
            if len(roots_possible) == 1 and cluster_num != 2:
                res_index = 0
                true_res_index = roots_possible[res_index]
                cluster_res_index = true_mapping_dict[true_res_index]
            else:  
                if cluster_num == 2:
                    roots_possible = X_se
                    
                print("possible roots: ", roots_possible)
                pair_cumulant_matrix = np.zeros((len(roots_possible), len(roots_possible)))
                pair_cumulant_list = []
                pair_cumulant_list_2 = []
                mapping_dict = {roots_possible[i]: i for i in range(len(roots_possible))}
                
                # 计算组内cumulant
                print("within")
                for i in roots_possible:
                    j = cluster_within_map_dict[i]
                    print([i, j])
                    
                    if j == -1:
                        pair_cumulant_list.append(-1)
                        pair_cumulant_list_2.append(-1)
                    else:    
                        pair_data = np.array([e_tilde_dict[i], data[:, j]]).T
                        ### trash version required by prof ###
                        # is_one_confounder = self.test_one_confounder_schkoda(pair_data)
                        ######################################
                        is_one_confounder = test_one_confounder_sixth_robust(pair_data, self._one_latent_tol)
                        
                        if is_one_confounder:
                            print("one confounder")
                            self.highest_l = 1
                            self.constraints = {l: get_constraints_for_l_latents(l) for l in range(self.highest_l+1)}
                            self.highest_order = calculate_orders_needed(1)[1]
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
                            
                            if i in self.directed_edge_within_observed:
                                self.directed_edge_within_observed[i].append(j)
                            else:
                                self.directed_edge_within_observed[i] = [j]

                print(mapping_dict)
                print(pair_cumulant_matrix)
                
                # 计算组间cumulant
                print("between")
                self.highest_l = 1
                self.highest_order = calculate_orders_needed(1)[1]
                
                for index_pair in permutations(roots_possible, 2):                
                    i, j = index_pair
                    print([i, j])
                    pair_data = np.array([e_tilde_dict[i], data[:, j]]).T
                    pair_cumulant_matrix[mapping_dict[j]][mapping_dict[i]] = self.estimate_latent_cumulants_1(pair_data)[1]

                pair_cumulant_matrix = np.array(pair_cumulant_matrix)
                print("pair_cumulant_matrix: ", pair_cumulant_matrix)
                
                result_list = []
                result_list_2 = []
                result_mean_list = []
                result_mean_list_2 = []
                
                for i in range(len(roots_possible)):
                    if pair_cumulant_list[i] == -1:
                        nonzero_list = [pair_cumulant_matrix[j][i] for j in range(len(pair_cumulant_matrix)) if i != j]
                        result_list.append(np.var(nonzero_list))
                        result_mean_list.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list[i])
                        print(temp_list)
                        nonzero_list = [temp_list[j] for j in range(len(temp_list)) if i != j]
                        result_list.append(np.var(nonzero_list))
                        result_mean_list.append(np.mean(nonzero_list))
                        
                    if pair_cumulant_list_2[i] == -1:
                        nonzero_list = [pair_cumulant_matrix[j][i] for j in range(len(pair_cumulant_matrix)) if i != j]
                        result_list_2.append(np.var(nonzero_list))
                        result_mean_list_2.append(np.mean(nonzero_list))
                    else:
                        temp_list = np.append(pair_cumulant_matrix[:, i], pair_cumulant_list_2[i])
                        print(temp_list)
                        nonzero_list = [temp_list[j] for j in range(len(temp_list)) if i != j]
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

                # 合并处理
                # true_res_index = roots_possible[res_index]
                # cluster_res_index = true_mapping_dict[true_res_index]
                # merge_cluster_list = []
                
                next_roots = [roots_possible[res_index]]
                for i in range(len(result_list)):
                    if i != res_index:
                        if np.abs(result_list[i] - result_list[res_index]) <= self._merge_threshold_next:
                            # merge_cluster_list.append(true_mapping_dict[roots_possible[i]])
                            next_roots.append(roots_possible[i])
                true_res_index = np.min(next_roots)
                cluster_res_index = true_mapping_dict[true_res_index]
                merge_cluster_list = []
                for i in next_roots:
                    if i != true_res_index:
                        merge_cluster_list.append(true_mapping_dict[i])

                merge_cluster_list.sort(reverse=True)
                for i in merge_cluster_list:
                    self.clusters[cluster_res_index] += self.clusters[i]
                    self.clusters.pop(i)
                    self.D_record.pop(self.X_se_now[i])
                    self.X_se_now.pop(i)
                    if i < cluster_res_index:
                        cluster_res_index -= 1
                        
                print("res_index: ", res_index)
                print("cluster_res_index: ", cluster_res_index)
                
            self.ordered_cluster.append(sorted(self.clusters[cluster_res_index]))
            self.X_se_now.pop(cluster_res_index)
            self.clusters.pop(cluster_res_index)
            
            now_cluster_index = len(self.ordered_cluster)-1
            e_tilde = self.remove_all_influence(true_res_index)
            
            for x in self.X_se_now:
                if x != true_res_index:
                    self.D_record[x]["A_list"].append(now_cluster_index)
                    self.D_record[x]["X_list"].append(true_res_index)
                    pho = compute_new_e(data[:, x], e_tilde)[0]
                    self.D_record[x]["S_list"].append(e_tilde)
                    self.D_record[x]["P_list"].append(pho)
                    
    def remove_redundant_edge(self, X_se_reverse):
        print(15*"=", "remove", "="*15)
        count = 0
        cluster_num = len(self.ordered_cluster)
        self.latent_adjmatrix = np.zeros((cluster_num, cluster_num), dtype=float)
        
        if cluster_num == 2:
            index_now = X_se_reverse[0]
            X_now = self.X[:, index_now].copy()  # 浅拷贝足够
            previous_ancestor = self.D_record[index_now]["S_list"][0]
            pho = compute_new_e(X_now, previous_ancestor)[0]
            self.latent_adjmatrix[1][0] = pho
            return self.latent_adjmatrix
        
        while cluster_num - count >= 2:
            print("X_se_reverse: ", X_se_reverse)
            index_now = X_se_reverse[count]
            print("index now: ", index_now)
            X_now = self.X[:, index_now].copy()  # 浅拷贝足够
            
            ancester_index_list = self.D_record[index_now]["X_list"][::-1]
            ancester_list = self.D_record[index_now]["S_list"][::-1]
            ancester_list_num = len(ancester_list)
            
            for index_ancestor in range(ancester_list_num):
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
                    # if np.fabs(pho_1) < 0.5:
                    if pho_1 < 0.5:
                    # if np.fabs(pho_1) < 0.01:
                        pho_1 = 0.0
                        self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = 0.0
                    else:
                        self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = pho_1
                    X_now -= pho_1 * self.X[:, X_index]
            count += 1

    def remove_redundant_edge_bootstrap(self, X_se_reverse):
        print(15*"=", "remove", "="*15)
        count = 0
        cluster_num = len(self.ordered_cluster)
        self.latent_adjmatrix = np.zeros((cluster_num, cluster_num), dtype=float)
        
        if cluster_num == 2:
            index_now = X_se_reverse[0]
            X_now = self.X[:, index_now].copy()  # 浅拷贝足够
            previous_ancestor = self.D_record[index_now]["S_list"][0]
            pho = compute_new_e(X_now, previous_ancestor)[0]
            self.latent_adjmatrix[1][0] = pho
            return self.latent_adjmatrix
        
        while cluster_num - count >= 2:
            print("X_se_reverse: ", X_se_reverse)
            index_now = X_se_reverse[count]
            print("index now: ", index_now)
            X_now = self.X[:, index_now].copy()  # 浅拷贝足够
            
            ancester_index_list = self.D_record[index_now]["X_list"][::-1]
            ancester_list = self.D_record[index_now]["S_list"][::-1]
            ancester_list_num = len(ancester_list)
            
            for index_ancestor in range(ancester_list_num):
                X_index = ancester_index_list[index_ancestor]
                print("ancestor index now: ", X_index)
                previous_ancestor = ancester_list[index_ancestor]
                        
                print("[", index_now, " -> ", X_index, "]")
                is_zero = is_coefficient_zero(X_now, previous_ancestor, self._ind_alpha)
                
                if is_zero:
                    self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = 0.0
                else:
                    pho_1 = 0.0
                    is_zero_bootstrap = is_coefficient_zero_bootstrap(X_now, previous_ancestor, B=1000)
                    
                    is_zero_bootstrap = False  # for test
                    
                    if is_zero_bootstrap:
                        pho_1 = 0.0
                        self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = 0.0
                        
                    else:
                        pho_1 = compute_new_e(X_now, previous_ancestor)[0]
                        self.latent_adjmatrix[cluster_num - count - 1][ancester_list_num - index_ancestor - 1] = pho_1
                    
                    X_now -= pho_1 * self.X[:, X_index]
                    print("pho: ", pho_1)
            count += 1

    def remove_all_influence(self, j):
        X_j = self.X[:, j]
        S_list = self.D_record[j]["S_list"]
        P_list = self.D_record[j]["P_list"]

        e_tilde_j_s = copy.deepcopy(X_j)
        
        for index in range(len(P_list)):
            e_tilde_j_s -= S_list[index] * P_list[index]

        return e_tilde_j_s
    

    def test_one_confounder_schkoda(self, data):
        """测试是否存在一个混杂变量的robust版本"""
        # indices_cum_4_2 = [0, 0, 0, 1]
        # indices_cum_2_4 = [0, 1, 1, 1]
        # indices_cum_3_3 = [0, 0, 1, 1]
        
        data_centered = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)

        confounders = self.find_all_confounders(data_centered)

        if confounders[0][1] == 1 and confounders[1][0] == 1:
            return True
        return False
    




def set_partitions(elements: List) -> Iterator[List[List]]:
    """递归生成集合的所有划分"""
    if not elements:
        yield []
        return
        
    if len(elements) == 1:
        yield [elements]
        return
    
    first = elements[0]
    rest = elements[1:]
    
    for partition in set_partitions(rest):
        for i in range(len(partition)):
            new_partition = []
            for j, block in enumerate(partition):
                if i == j:
                    new_partition.append([first] + block)
                else:
                    new_partition.append(block[:])
            yield new_partition
        
        yield [[first]] + [block[:] for block in partition]


def rank_transform_1d(x, method="average", uniform="cdf"):
    """
    对 1D 向量做秩/分位变换。
    参数
    ----
    x : array-like, shape (n,)
        输入数据，可含 NaN（NaN 将在输出中保留为 NaN）。
    method : {"average","min","max","dense","first"}
        秩的并列（ties）处理方式，对应 pandas.Series.rank 的 method。
    uniform : {"cdf","midrank"}
        - "cdf": u = rank / (n + 1)
        - "midrank": u = (rank - 0.5) / n

    返回
    ----
    u : np.ndarray, shape (n,)
        归一化到 (0,1) 或 [0,1) 以内的分位值；NaN 保留。
    """
    s = pd.Series(x.T[0], dtype="float64")  # 保留 NaN
    n = s.notna().sum()
    if n == 0:
        return s.to_numpy()  # 全 NaN 直接返回

    ranks = s.rank(method=method, na_option="keep")  # NaN 不参与排名，保持 NaN
    if uniform == "cdf":
        u = ranks / (n + 1.0)
    elif uniform == "midrank":
        u = (ranks - 0.5) / n
    else:
        raise ValueError('uniform must be "cdf" or "midrank"')
    return u.to_numpy()
    


def cumulant(data, indices):
    """计算高阶交叉累积量"""
    cumulant_value = 0.0
    indices_list = list(indices)
    
    for partition in set_partitions(indices_list):
        prod_moment = 1.0
        for block in partition:
            vals = np.prod(data[:, block], axis=1)
            prod_moment *= vals.mean()
        
        weight = math.factorial(len(partition) - 1) * ((-1) ** (len(partition) - 1))
        cumulant_value += weight * prod_moment
        
    return cumulant_value
    
    
def test_one_confounder_sixth_robust(data, one_latent_tol):
    """测试是否存在一个混杂变量的robust版本"""
    indices_cum_4_2 = [0, 0, 0, 1]
    indices_cum_2_4 = [0, 1, 1, 1]
    indices_cum_3_3 = [0, 0, 1, 1]
    
    data_centered = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
    
    cum_4_2 = cumulant(data_centered, indices_cum_4_2)
    cum_2_4 = cumulant(data_centered, indices_cum_2_4)
    cum_3_3 = cumulant(data_centered, indices_cum_3_3)
    
    left_side = cum_4_2 * cum_2_4
    right_side = cum_3_3**2
    
    abs_left = abs(left_side)
    abs_right = abs(right_side)
    max_abs = max(abs_left, abs_right)
    
    if max_abs < 1e-12:
        rel_err = 0.0
    else:
        rel_err = abs(left_side - right_side) / max_abs
    
    tol = one_latent_tol
    
    print(f"cum(X₁⁴X₂²) × cum(X₁²X₂⁴) = {left_side}")
    print(f"cum(X₁³X₂³)² = {right_side}")
    print(f"绝对误差: {left_side - right_side}")
    print(f"相对误差: {rel_err}")
    print(f"宽容度: {tol}")
    
    return min(rel_err, abs(left_side - right_side)) < tol
    # return rel_err < tol






def compute_new_e(X_i, X_j):
    """计算 X_i - rho*X_j"""
    data = np.column_stack((X_i, X_j))  # 避免转置操作

    # 要用四阶，但是normallog用一下三阶
    cum_iij = cumulant(data, [0, 0, 1, 1])
    cum_ijj = cumulant(data, [0, 1, 1, 1])

    # cum_iij = cumulant(data, [0, 0, 1])
    # cum_ijj = cumulant(data, [0, 1, 1])

    rho = cum_iij / cum_ijj
    new_e = X_i - rho * X_j

    return rho, new_e


def is_coefficient_zero(X_i, X_j, ind_alpha):
    """判断系数是否为零"""
    N = min(len(X_i), 2000)
    temp_data = np.column_stack((X_i, X_j))
    
    ###############
    data_centered = (temp_data - temp_data.mean(axis=0)) / temp_data.std(axis=0)
    _, p_value = Hsic(compute_kernel="laplacian").test(data_centered[:N, [0]], data_centered[:N, [1]])
    ###############

    # data_centered = (temp_data - temp_data.mean(axis=0)) / temp_data.std(axis=0)
    # _, p_value = Hsic(compute_kernel="rbf").test(rank_transform_1d(temp_data[:N, [0]]), rank_transform_1d(temp_data[:N, [1]]))
    
    print("hsic: ", p_value)
    
    return p_value >= ind_alpha

def is_coefficient_zero_bootstrap(X_i, X_j, B=2500):
    """判断系数是否为零"""
    threshold = 0.05
    threshold_time_100 = 100 * (threshold)
    n = len(X_i)
    a_bootstrap = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        X_i_b = X_i[idx] # now
        X_j_b = X_j[idx] # previous_ancestor
        pho = compute_new_e(X_i_b, X_j_b)[0]
        if not np.isnan(pho):
            a_bootstrap.append(pho)
    percentile_025 = np.percentile(a_bootstrap, threshold_time_100/2)
    percentile_975 = np.percentile(a_bootstrap, 100-threshold_time_100/2)
    
    print("percentile_025, percentile_975: ", percentile_025, percentile_975)

    if percentile_025 < 0.0 and percentile_975 > 0.0:
        return True    
    return False
    # return np.array(a_bootstrap)


    # for _ in range(B):
        # N = min(len(X_i), 2000)
        # temp_data = np.column_stack((X_i, X_j))
    # data_centered = (temp_data - temp_data.mean(axis=0)) / temp_data.std(axis=0)
    # _, p_value = Hsic(compute_kernel="laplacian").test(data_centered[:N, [0]], data_centered[:N, [1]])
    # print("hsic: ", p_value)
    
    # return p_value >= ind_alpha


def fisher_test(pvals):
    """Fisher检验"""
    pvals = [max(pval, 1e-5) for pval in pvals]
    fisher_stat = -2.0 * np.sum(np.log(pvals))
    return 1 - chi2.cdf(fisher_stat, 2 * len(pvals))
import sys
sys.path.append("./LaHiCaSl")
import time
import numpy as np
import utils
import pandas as pd
import lingam
import ReLVLiNGAM
import LSLiNGAM as LSL
import LSLiNGAM_claude as LSL_C
from lingam.hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
from scipy.stats import pearsonr
from causallearn_local.search.HiddenCausal.GIN.GIN import GIN, GIN_MI
import evaluate

import LaHiCaSl.LaHiCaSl as Xie

# 配置常量
EXPERIMENT_CONFIG = {
    # "exp_num": 100,
    "exp_num": 5,
    # "experiment_set": [1000, 2000, 4000, 8000, 16000],
    # "experiment_set": [1000],
    "experiment_set": [50, 100, 200, 400, 800],
    # "setting_case_set": ["case1", "case2", "case3", "case4", "case5", "case6"],  # 可根据需要调整
    "setting_case_set": ["case5", "case6"],  # 可根据需要调整
    "para_list": [0.05, 0.001, [0.001, 0.005], 0.001, 0.01],
    # "para_list": [[0.01, 0.05, 0.1, 0.2, 0.3], [0.001, 0.01, 0.1], [0.001, 0.005], 0.001, 0.01],
    "columns_res": [
        "case", "sample_size", "ind_alpha", 'one_latent_tol', 'singular_threshold', 
        'merge_threshold_first', 'merge_threshold_next', 
        'is_cluster_correct_count', 'is_latent_edge_correct_count', "is_observed_edge_correct_count", 'is_totally_correct_count',
        'cluster_PRE', 'cluster_REC', 'cluster_F1', 'edge_PRE', 'edge_REC', 'edge_F1',
        'observed_edge_PRE', 'observed_edge_REC', 'observed_edge_F1', 'RMSE_mean', 'RMSE_std',
        'RMSE_correct_mean', 'RMSE_correct_std'
    ]
}

# 实验设置配置
SETTINGS = {
    "case1": {
        "matrix": np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 1, 0]
        ], dtype=float),
        "hidden_num": 1,
        "cluster": [[0, 1, 2]],
        "observed_edge": {1: [2]},
        "high_l": 1,
        "case_name": "case1"
    },
    "case2": {
        "matrix": np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ], dtype=float),
        "hidden_num": 1,
        "cluster": [[0, 1, 2]],
        "observed_edge": {0: [1, 2], 1: [2] },
        "high_l": 1,
        "case_name": "case2"
    },
    "case3": {
        "matrix": np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
        ], dtype=float),
        "hidden_num": 2,
        "cluster": [[0], [1, 2]],
        "observed_edge": {1: [2]},
        "high_l": 1,
        "case_name": "case3"
    },
    "case4": {
        "matrix": np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
        ], dtype=float),
        "hidden_num": 2,
        "cluster": [[0], [1, 2, 3]],
        "observed_edge": {2: [3]},
        "high_l": 1,
        "case_name": "case4"
    },
    "case5": {
        "matrix": np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
        ], dtype=float),
        "hidden_num": 3,
        "cluster": [[0], [1], [2, 3]],
        "observed_edge": {2: [3]},
        "high_l": 1,
        "case_name": "case5"
    },
    "case6": {
        "matrix": np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
        ], dtype=float),
        "hidden_num": 3,
        "cluster": [[0], [1], [2, 3]],
        "observed_edge": {2: [3]},
        "high_l": 1,
        "case_name": "case6"
    },
}


class ExperimentRunner:
    """实验运行器类，封装实验逻辑"""
    
    def __init__(self, config=None):
        self.config = config or EXPERIMENT_CONFIG
        self.para_list = self.config["para_list"]
        # 存储每个case的B_matrix列表
        self.case_b_matrices = {}
        
    def generate_b_matrices_for_case(self, case_config, exp_num):
        """为一个case生成固定数量的B_matrix"""
        matrix = case_config["matrix"]
        hidden_num = case_config["hidden_num"]
        node_num = len(matrix)
        
        b_matrices = []
        for exp in range(exp_num):
            # 生成一个临时的扰动项用于生成B_matrix
            # temp_disturbances = utils.generate_disturbances("lognormal", node_num, 1000, [])
            B_matrix = utils.generate_coefficient_matrix(matrix, hidden_num)
            b_matrices.append(B_matrix)
        
        return b_matrices
    
    def generate_experiment_data(self, case_config, sample_size, B_matrix):
        """使用给定的B_matrix生成实验数据"""
        matrix = case_config["matrix"]
        hidden_num = case_config["hidden_num"]
        node_num = len(matrix)
        
        disturbances = utils.generate_disturbances("lognormal", node_num, sample_size, [])
        X, X_hidden, X_observed, Mixing, Mixing_hidden, Mixing_observed = utils.generate_data(
            disturbances, B_matrix, hidden_num
        )
        
        return X_observed
    
    def run_lslingam_experiment(self, X_observed, case_config):
        """运行LSLiNGAM实验"""
        singular_threshold = 0
        if case_config["case_name"] in ["case1", "case2", "case3"]:
            singular_threshold = self.para_list[2][0]
        else:
            singular_threshold = self.para_list[2][1]
        # 使用LSLiNGAM进行实验
        model = LSL_C.LSLiNGAM(
            X_observed, 
            case_config["high_l"],
            ind_alpha=self.para_list[0],
            one_latent_tol=self.para_list[1],
            # singular_threshold=self.para_list[2],
            singular_threshold=singular_threshold,
            merge_threshold_first=self.para_list[3],
            merge_threshold_next=self.para_list[4]
        )
        model.fit()

        # print("LSLiNGAM Results:")
        # print("Cluster Estimates:", model.ordered_cluster)
        # print("Latent Graph Estimates:", model.latent_adjmatrix)
        print("Observed Edge Estimates:", model.directed_edge_within_observed)
        return {
            "cluster_est": model.ordered_cluster,
            "G_est": model.latent_adjmatrix,
            "observed_edge_est": model.directed_edge_within_observed
        }
    
    def run_gin_experiments(self, X_observed):
        """运行GIN和GIN_MI实验"""
        # GIN实验
        # print("sample size:", len(X_observed))
        N = min(5000, len(X_observed))  # 限制样本大小以避免内存问题
        G_gin, K_gin, latent_adjmatrix_gin = GIN(X_observed[:N, :], indep_test_method='hsic', alpha=self.para_list[0])
        gin_results = {
            "cluster_est": K_gin,
            "G_est": latent_adjmatrix_gin,
            "observed_edge_est": {}
        }
        
        # GIN_MI实验
        G_gin_mi, K_gin_mi, latent_adjmatrix_gin_mi = GIN_MI(X_observed[:N, :])
        gin_mi_results = {
            "cluster_est": K_gin_mi,
            "G_est": latent_adjmatrix_gin_mi,
            "observed_edge_est": {}
        }

        X_pd = pd.DataFrame(X_observed[:N, :])
        latent_adjmatrix_gin_impure, K_gin_impure, observed_edge_impure = Xie.Latent_Hierarchical_Causal_Structure_Learning(X_pd, alpha=self.para_list[0])
        gin_impure_results = {
            "cluster_est": K_gin_impure,
            "G_est": latent_adjmatrix_gin_impure,
            "observed_edge_est": observed_edge_impure
        }
        return gin_results, gin_mi_results, gin_impure_results

    def run_relvlingam_experiments(self, X_observed, case_config):
        lingam_R = ReLVLiNGAM.ReLVLiNGAM(X_observed, case_config["high_l"])
        lingam_R.fit(X_observed)
        B_hat = lingam_R.B
        
        print("ReLVLiNGAM Estimated Topological Order:")
        # print(lingam_R.topological_order)
        print(lingam_R.topological_order)
        # B_hat = np.where(B_hat < 0.5, 0, B_hat)
        print("ReLVLiNGAM Estimated B matrix:")
        print(B_hat)
        observed_number = B_hat.shape[0]
        latent_num = B_hat.shape[1] - B_hat.shape[0]
        observed_edges_dict = dict()
        cluster_est = list()
        if case_config["case_name"] == "case1" or case_config["case_name"] == "case2":
            if latent_num == case_config["hidden_num"]:
                cluster_est = [[i for i in range(B_hat.shape[0])]]
                observed_mixing_matrix = B_hat[:, : observed_number]
                for i in range(observed_mixing_matrix.shape[0]):
                    for j in range(observed_mixing_matrix.shape[1]):
                        if i != j:
                            if observed_mixing_matrix[i][j] != 0:
                                if j in observed_edges_dict.keys():
                                    observed_edges_dict[j].append(i)
                                else:
                                    observed_edges_dict[j] = [i]
        print("ReLVLiNGAM Estimated Observed Edges:")
        print(observed_edges_dict)
        return {
            "cluster_est": cluster_est,
            "G_est": np.array([[0.0]]),
            "observed_edge_est": observed_edges_dict
        }
        # return [latent_num, observed_edges_dict]
    #     return [
    #     is_cluster_correct_count, is_latent_edge_correct_count, is_totally_correct_count,
    #     cluster_PRE, cluster_REC, cluster_F1,
    #     edge_PRE, edge_REC, edge_F1,
    #     observed_edge_PRE, observed_edge_REC, observed_edge_F1,
    #     RMSE_mean, RMSE_std
    # ]

            # print(f"Warning: Estimated latent number {latent_num} exceeds expected {case_config['hidden_num']}.")
        # if latent_num == case_config["hidden_num"]:
        #     latent_num = 0
        # return latent_num

    def evaluate_results(self, results, case_config, B_matrix, is_proposed=True):
        """评估结果"""
        hidden_num = case_config["hidden_num"]
        return evaluate.evaluate_one(
            results["G_est"],
            results["cluster_est"],
            results["observed_edge_est"],
            B_matrix[:hidden_num, :hidden_num],
            case_config["cluster"],
            case_config["observed_edge"],
            is_proposed=is_proposed
        )
    
    # def evaluate_latent_number(self, results_lists, case_config):
    #     """评估潜在变量数量"""
    #     # 计算每个实验的潜在变量数量
    #     # latent_counts = [len(results["cluster_est"]) for results in results_lists]
    #     # 返回平均潜在变量数量和标准差
    #     latent_correct_count = sum(1 for count in results_lists if count == case_config["hidden_num"])
    #     observed_edges_dict_true = case_config["observed_edge"]
    #     return [
    #         latent_correct_count, 0, is_totally_correct_count,
    #         cluster_PRE, cluster_REC, cluster_F1,
    #         0, 0, 0,
    #         observed_edge_PRE, observed_edge_REC, observed_edge_F1,
    #         0, 0
    #     ]
    #     # return latent_correct_count

    
    def run_single_case_experiment(self, case_name, case_config, sample_size, exp_num):
        """运行单个案例的所有实验"""
        print(f"Running experiments for {case_name} with sample size {sample_size}")
        
        # 如果还没有为这个case生成B_matrix，现在生成
        if case_name not in self.case_b_matrices:
            self.case_b_matrices[case_name] = self.generate_b_matrices_for_case(case_config, exp_num)
        
        # 使用已生成的B_matrix列表
        b_matrices = self.case_b_matrices[case_name]
        
        results_lists = {"lsl": [], "gin": [], "gin_mi": [], "gin_impure": [], "relvlingam": []}
        
        for exp in range(exp_num):
            print(f"  Experiment {exp + 1}/{exp_num}")
            
            # 使用第exp个B_matrix生成数据
            B_matrix = b_matrices[exp]
            X_observed = self.generate_experiment_data(case_config, sample_size, B_matrix)
            
            # LSLiNGAM实验
            lsl_results = self.run_lslingam_experiment(X_observed, case_config)
            lsl_score = self.evaluate_results(lsl_results, case_config, B_matrix, is_proposed=True)
            results_lists["lsl"].append(lsl_score)
            
            # # GIN实验
            # gin_results, gin_mi_results, gin_impure_results = self.run_gin_experiments(X_observed)
            # gin_score = self.evaluate_results(gin_results, case_config, B_matrix, is_proposed=False)
            # gin_mi_score = self.evaluate_results(gin_mi_results, case_config, B_matrix, is_proposed=False)
            # gin_impure_score = self.evaluate_results(gin_impure_results, case_config, B_matrix, is_proposed=False)

            # results_lists["gin"].append(gin_score)
            # results_lists["gin_mi"].append(gin_mi_score)
            # results_lists["gin_impure"].append(gin_impure_score)

            # # ReLVLiNGAM实验
            # if case_name == "case1" or case_name == "case2":
            #     relvlingam_result = self.run_relvlingam_experiments(X_observed, case_config)
            #     relvlingam_score = self.evaluate_results(relvlingam_result, case_config, B_matrix, is_proposed=False)
            #     # results_lists["relvlingam"].append(relvlingam_latent_num)
            #     results_lists["relvlingam"].append(relvlingam_score)

        # 计算汇总结果
        return {
            "lsl": evaluate.evaluate_repeat(results_lists["lsl"]),
            # "gin": evaluate.evaluate_repeat(results_lists["gin"]),
            # "gin_mi": evaluate.evaluate_repeat(results_lists["gin_mi"]),
            # "gin_impure": evaluate.evaluate_repeat(results_lists["gin_impure"]),
            # # "relvlingam": [self.evaluate_latent_number(results_lists["relvlingam"], case_config)]
            # "relvlingam": evaluate.evaluate_repeat(results_lists["relvlingam"])
        }
    
    def run_all_experiments(self):
        """运行所有实验"""
        # score_records = {"lsl": {}, "gin": {}, "gin_mi": {}, "gin_impure": {}, "relvlingam": {}}
        score_records = {"lsl": {}}
        
        # 初始化记录结构
        for method in score_records:
            for case_name in self.config["setting_case_set"]:
                score_records[method][case_name] = {}
        
        # 运行实验
        for case_name in self.config["setting_case_set"]:
            case_config = SETTINGS[case_name]
            
            # 为这个case生成B_matrix（只生成一次）
            if case_name not in self.case_b_matrices:
                self.case_b_matrices[case_name] = self.generate_b_matrices_for_case(
                    case_config, self.config["exp_num"]
                )
            
            for sample_size in self.config["experiment_set"]:
                case_results = self.run_single_case_experiment(
                    case_name, case_config, sample_size, self.config["exp_num"]
                )
                
                # 存储结果
                for method in score_records:
                    score_records[method][case_name][sample_size] = case_results[method]
        
        return score_records
    
    def save_results_to_excel(self, score_records):
        """保存结果到Excel文件"""
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
        # 为每种方法生成DataFrame
        method_dataframes = {}
        
        # for method_name, method_key in [("LSLiNGAM", "lsl"), ("GIN", "gin"), ("GIN_MI", "gin_mi"), ("GIN_Impure", "gin_impure"), ("ReLVLiNGAM", "relvlingam")]:
        for method_name, method_key in [("LSLiNGAM", "lsl")]:
            array_res = []
            case_count = 0
            
            for case_name in self.config["setting_case_set"]:
                case_count += 1
                for sample_size in self.config["experiment_set"]:
                    result_row = [case_count, sample_size] + self.para_list + \
                                list(score_records[method_key][case_name][sample_size])
                    array_res.append(result_row)
            
            if method_name == "ReLVLiNGAM":
                # method_dataframes[method_name] = pd.DataFrame(array_res, columns=self.config["columns_res"][:8])
                method_dataframes[method_name] = pd.DataFrame(array_res, columns=self.config["columns_res"])
            else:
                method_dataframes[method_name] = pd.DataFrame(array_res, columns=self.config["columns_res"])
            # 保存到文件
            filename = f"{method_name}_res_fixed_B_{formatted_time}.xlsx"
            method_dataframes[method_name].to_excel(filename, index=False)
            print(f"Results saved to {filename}")
        
        return method_dataframes

    def save_b_matrices_info(self):
        """保存B_matrix信息以便后续分析"""
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
        b_matrix_info = {}
        for case_name, b_matrices in self.case_b_matrices.items():
            b_matrix_info[case_name] = {
                "count": len(b_matrices),
                "shapes": [b_matrix.shape for b_matrix in b_matrices],
                "first_matrix": b_matrices[0].tolist() if b_matrices else None
            }
        
        # 保存到文件
        import json
        filename = f"B_matrices_info_{formatted_time}.json"
        with open(filename, 'w') as f:
            json.dump(b_matrix_info, f, indent=2)


def main():
    """主函数"""
    print("Starting experiments...")
    
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 运行所有实验
    score_records = runner.run_all_experiments()
    
    # 保存结果
    dataframes = runner.save_results_to_excel(score_records)
    
    # 保存B_matrix信息
    runner.save_b_matrices_info()
    
    # 打印结果概览
    print("\nExperiment completed!")
    for method_name, df in dataframes.items():
        print(f"\n{method_name} Results Summary:")
        print(df.head())


if __name__ == '__main__':
    np.random.seed(1966)  # 设置随机种子以确保结果可复现
    main()
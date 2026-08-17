import sys
sys.path.append("./LaHiCaSl")
import time
import numpy as np
import utils
# import utils_nearGaussian_02
# import utils_nearGaussian_05
import pandas as pd
# import LSLiNGAM as LSL
import LSLiNGAM_oracle_merge as LSL_C
import evaluate

# 配置常量
EXPERIMENT_CONFIG = {
    "exp_num": 100,
    # "exp_num": 2,
    "experiment_set": [1000, 2000, 4000, 8000, 16000],
    # "experiment_set": [16000],
    # "setting_case_set": ["case1", "case2", "case3", "case4", "case5", "case6"],  # 可根据需要调整
    # "setting_case_set": ["case5", "case6"],  # 可根据需要调整
    "setting_case_set": ["case4"],  # 可根据需要调整
    # "setting_case_set": ["case1"],  # 可根据需要调整
    "para_list": [0.05, 0.001, [0.001, 0.005], 0.001, 0.01],
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
    """实验运行器类，封装实验逻辑（仅运行LSLiNGAM）"""
    
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
        # disturbances = utils_nearGaussian_05.generate_disturbances("lognormal", node_num, sample_size, [])
        # disturbances = utils_nearGaussian_02.generate_disturbances("lognormal", node_num, sample_size, [])
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
        # model = LSL_C.LSLiNGAM(
        #     X_observed, 
        #     case_config["high_l"],
        #     ind_alpha=self.para_list[0],
        #     one_latent_tol=self.para_list[1],
        #     # singular_threshold=self.para_list[2],
        #     singular_threshold=singular_threshold,
        #     merge_threshold_first=self.para_list[3],
        #     merge_threshold_next=self.para_list[4]
        # )
        model = LSL_C.LSLiNGAM(
            X_observed,
            case_config["high_l"],
            ind_alpha=self.para_list[0],
            one_latent_tol=self.para_list[1],
            singular_threshold=singular_threshold,
            merge_threshold_first=self.para_list[3],
            merge_threshold_next=self.para_list[4],
            # oracle_clusters=case_config["cluster"],  # 传入真实clusters
            oracle_clusters=[[0], [1], [2, 3]],
            oracle_observed_edges=case_config["observed_edge"]
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

    def run_single_case_experiment(self, case_name, case_config, sample_size, exp_num):
        """运行单个案例的所有实验"""
        print(f"Running experiments for {case_name} with sample size {sample_size}")
        
        # 如果还没有为这个case生成B_matrix，现在生成
        if case_name not in self.case_b_matrices:
            self.case_b_matrices[case_name] = self.generate_b_matrices_for_case(case_config, exp_num)
        
        # 使用已生成的B_matrix列表
        b_matrices = self.case_b_matrices[case_name]
        
        results_lists = {"lsl": []}
        
        for exp in range(exp_num):
            print(f"  Experiment {exp + 1}/{exp_num}")
            
            # 使用第exp个B_matrix生成数据
            B_matrix = b_matrices[exp]
            X_observed = self.generate_experiment_data(case_config, sample_size, B_matrix)
            
            # LSLiNGAM实验
            lsl_results = self.run_lslingam_experiment(X_observed, case_config)
            lsl_score = self.evaluate_results(lsl_results, case_config, B_matrix, is_proposed=True)
            results_lists["lsl"].append(lsl_score)

        # 计算汇总结果
        return {
            "lsl": evaluate.evaluate_repeat(results_lists["lsl"])
        }
    
    def run_all_experiments(self):
        """运行所有实验"""
        score_records = {"lsl": {}}
        
        # 初始化记录结构
        for method in score_records:
            for case_name in self.config["setting_case_set"]:
                score_records[method][case_name] = {}
        
        # 运行实验
        # np.random.seed(2025)
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
        
        for method_name, method_key in [("LSLiNGAM", "lsl")]:
            array_res = []
            case_count = 0
            
            for case_name in self.config["setting_case_set"]:
                case_count += 1
                for sample_size in self.config["experiment_set"]:
                    result_row = [case_count, sample_size] + self.para_list + \
                                list(score_records[method_key][case_name][sample_size])
                    array_res.append(result_row)
            
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
    print("Starting experiments (LSLiNGAM only)...")
    
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

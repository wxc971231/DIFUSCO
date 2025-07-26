#!/usr/bin/env python3
"""使用DIFUSCO TSP1000 checkpoint生成TSP数据的独立脚本"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset

# 添加difusco模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'difusco'))

from difusco.pl_tsp_model import TSPModel
from difusco.utils.tsp_utils import TSPEvaluator
from difusco.utils.diffusion_schedulers import InferenceSchedule
from environment.used.BaseEnv_COP import RawData

def load_model(args, device):
    """加载TSP模型，基于train.py的加载方式"""
    
    # 直接从checkpoint加载模型，让PyTorch Lightning处理参数
    model = TSPModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        param_args=args
    ).to(device)
    
    return model

def generate_tsp_solution_with_model(model, points, device):
    """使用TSPModel实例生成TSP解，完全复制test_step逻辑"""
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # 准备数据格式，与test_step保持一致
        if model.sparse:
            # 稀疏图处理
            from sklearn.neighbors import KDTree
            from torch_geometric.data import Data as GraphData
            
            sparse_factor = model.args.sparse_factor
            kdt = KDTree(points, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)
            
            edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0).to(device)
            
            # 创建图数据
            x = torch.from_numpy(points).float().to(device)
            edge_attr = torch.zeros(edge_index.shape[1], device=device)
            graph_data = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            # 模拟batch数据
            real_batch_idx = torch.tensor([0], device=device)
            point_indicator = torch.zeros(1, device=device)
            edge_indicator = torch.zeros(edge_index.shape[1], device=device)
            gt_tour = torch.arange(points.shape[0], device=device)
            
            batch = (real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour)
            
            # 处理稀疏图数据
            route_edge_flags = graph_data.edge_attr
            points_tensor = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            points_tensor = points_tensor.reshape((-1, 2))
            edge_index = edge_index.reshape((2, -1))
            np_points = points
            np_edge_index = edge_index.cpu().numpy()
            
        else:
            # 密集图处理
            real_batch_idx = torch.tensor([0], device=device)
            points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
            adj_matrix = torch.zeros((1, points.shape[0], points.shape[0]), device=device)
            gt_tour = torch.arange(points.shape[0], device=device).unsqueeze(0)
            
            batch = (real_batch_idx, points_tensor, adj_matrix, gt_tour)
            np_points = points_tensor.cpu().numpy()[0]
            edge_index = None
            np_edge_index = None
        
        # 临时设置单次采样
        model.args.parallel_sampling = 1
        model.args.sequential_sampling = 1
        
        # 执行推理（完全复制test_step逻辑）
        xt = torch.randn_like(adj_matrix.float())
        
        if model.diffusion_type == 'gaussian':
            xt.requires_grad = True
        else:
            xt = (xt > 0).long()
        
        if model.sparse:
            xt = xt.reshape(-1)
        
        steps = model.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(
            inference_schedule=model.args.inference_schedule,
            T=model.diffusion.T, 
            inference_T=steps
        )
        
        # 扩散迭代
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)
            
            if model.diffusion_type == 'gaussian':
                xt = model.gaussian_denoise_step(
                    points_tensor, xt, t1, device, edge_index, target_t=t2
                )
            else:
                xt = model.categorical_denoise_step(
                    points_tensor, xt, t1, device, edge_index, target_t=t2
                )
        
        # 后处理
        if model.diffusion_type == 'gaussian':
            adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        else:
            adj_mat = xt.float().cpu().detach().numpy() + 1e-6
        
        # 提取tour
        from utils.tsp_utils import merge_tours, batched_two_opt_torch
        
        tours, _ = merge_tours(
            adj_mat, np_points, np_edge_index,
            sparse_graph=model.sparse,
            parallel_sampling=1
        )
        
        # 2-opt优化
        solved_tours, _ = batched_two_opt_torch(
            np_points.astype("float64"), 
            np.array(tours).astype('int64'),
            max_iterations=model.args.two_opt_iterations, 
            device=device
        )

        return solved_tours[0]

def main():
    parser = argparse.ArgumentParser(description='使用DIFUSCO TSP1000 checkpoint生成TSP数据')
    parser.add_argument('--data_path', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/tsp1000_categorical.ckpt',help='TSP1000 checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=1000,help='TSP节点数量')
    parser.add_argument('--num_samples', type=int, default=100,help='生成样本数量')
    parser.add_argument('--output_file', type=str, default='generated_tsp1000_data.txt',help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')
    parser.add_argument('--diffusion_type', type=str, default='gaussian')
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_schedule', type=str, default='linear')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)

    args = parser.parse_args()
    args.sparse_factor = 50
    args.inference_diffusion_steps = 50
    args.inference_schedule = 'cosine'
    args.diffusion_type = 'categorical'
    args.task = 'tsp'
    args.ckpt_path = "/data1/autoco/DIFUSCO/ckpt/tsp1000_categorical.ckpt"
    args.storage_path = "/data1/autoco/DIFUSCO/output"
    args.output_file = "/data1/autoco/DIFUSCO/output/generated_tsp1000_data.txt"

    args.num_nodes = 1000
    args.num_samples = 100
    
    # 加载模型
    device = torch.device(args.device)
    model = load_model(args, device)
    
    # 加载数据
    # loaded_dataset = TSPGraphDataset(
    #     data_file=os.path.join(args.storage_path,args.data_path),
    #     sparse_factor=args.sparse_factor,
    # )
    # points, tour_gt = loaded_dataset.get_example(i)

    # 生成数据
    seed = 42
    rng = np.random.default_rng(seed)
    points_all = rng.random([args.num_samples, args.num_nodes, 2])   # (num_samples, num_nodes, 2)

    # 开始求解
    raw_dataset = RawData(seed_list=[seed,], problem_list=[], answer_list=[], cost_list=[])
    for i in tqdm(range(args.num_samples), desc="生成进度"):
        try:
            points = points_all[i]
            tour = generate_tsp_solution_with_model(model, points, args.device)
            
            # 验证解的有效性
            if len(set(tour)) != args.num_nodes:
                print(f"警告: 样本 {i} 的解无效，跳过")
                continue
            
            # 计算解的成本
            evaluator = TSPEvaluator(points)
            cost = evaluator.evaluate(tour)

            # 记录数据
            raw_dataset.problem_list.append({'position': points})
            raw_dataset.answer_list.append(tour[:-1].tolist())
            raw_dataset.cost_list.append(cost)
                
        except Exception as e:
            print(f"生成样本 {i} 时出错: {e}")
            continue
    
    print(f"数据生成完成，保存到: {args.output_file}")


if __name__ == '__main__':
    main()
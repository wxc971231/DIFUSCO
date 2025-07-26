#!/usr/bin/env python3
"""基于gen_data.py的多GPU多进程TSP数据生成脚本"""

import os
import sys
import argparse
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue
from tqdm import tqdm
import time
import pickle
# 添加difusco模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'difusco'))

from difusco.pl_tsp_model import TSPModel
from difusco.utils.tsp_utils import TSPEvaluator
from environment.used.BaseEnv_COP import RawData
from gen_data import load_model, generate_tsp_solution_with_model
import queue 

def worker_process(gpu_id, process_id, args, samples_per_process, worker_seed, result_queue):
    """工作进程函数"""
    try:
        # 设置GPU
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        
        # 设置随机种子
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        print(f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}) started with seed {worker_seed}")
        
        # 加载模型
        model = load_model(args, device)
        if model is None:
            result_queue.put((process_id, None, f"Failed to load model on GPU {gpu_id}"))
            return
        
        # 生成数据
        rng = np.random.default_rng(worker_seed)
        points_all = rng.random([samples_per_process, args.num_nodes, 2])
        
        raw_dataset = RawData(seed_list=[worker_seed], problem_list=[], answer_list=[], cost_list=[])
        for i in range(samples_per_process):
            try:
                points = points_all[i]
                tour = generate_tsp_solution_with_model(model, points, device)
                
                # 验证解的有效性
                if len(set(tour)) != args.num_nodes:
                    print(f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}): 样本 {i} 的解无效，跳过")
                    continue
                
                # 计算解的成本
                evaluator = TSPEvaluator(points)
                cost = evaluator.evaluate(tour)

                # 记录数据
                raw_dataset.problem_list.append({'position': points})
                raw_dataset.answer_list.append(tour[:-1].tolist())
                raw_dataset.cost_list.append(cost)
                
                if (i + 1) % 10 == 0:
                    print(f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}) completed {i + 1}/{samples_per_process} samples, ave_cost {np.mean(raw_dataset.cost_list)}")
                    
            except Exception as e:
                print(f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}) error on sample {i}: {e}")
                continue
        
        result_queue.put((process_id, raw_dataset, None))
        print(f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}) completed all {len(raw_dataset.answer_list)} samples")
        
    except Exception as e:
        result_queue.put((process_id, None, f"Worker {process_id}(Seed{worker_seed}-GPU{gpu_id}) failed: {e}"))

def create_args():
    """创建参数对象"""
    args = argparse.Namespace()
    
    # 基本参数
    args.data_path = 'data/tsp/tsp50_test_concorde.txt'
    args.ckpt_path = "/data1/autoco/DIFUSCO/ckpt/tsp1000_categorical.ckpt"
    args.batch_size = 64
    args.num_nodes = 1000
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 42
    args.num_workers = 0
    args.fp16 = False
    args.use_activation_checkpoint = False
    
    # 模型参数
    args.diffusion_type = 'categorical'
    args.diffusion_schedule = 'linear'
    args.diffusion_steps = 1000
    args.inference_diffusion_steps = 50
    args.inference_schedule = 'cosine'
    args.inference_trick = "ddim"
    args.n_layers = 12
    args.hidden_dim = 256
    args.sparse_factor = 50
    args.aggregation = 'sum'
    args.two_opt_iterations = 1000
    args.task = 'tsp'
    args.storage_path = "/data1/autoco/DIFUSCO/output"
    
    return args

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Multi-Process TSP Solution Generation')
    parser.add_argument('--num_samples', type=int, default=1000, help='Total number of samples to generate')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--processes_per_gpu', type=int, default=2, help='Number of processes per GPU')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed')
    
    cmd_args = parser.parse_args()
    cmd_args.num_samples = 10000
    cmd_args.num_gpus = 6
    cmd_args.processes_per_gpu = 10
        
    if cmd_args.num_gpus > torch.cuda.device_count():
        print(f"Requested {cmd_args.num_gpus} GPUs but only {torch.cuda.device_count()} available")
        cmd_args.num_gpus = torch.cuda.device_count()
    
    total_processes = cmd_args.num_gpus * cmd_args.processes_per_gpu
    samples_per_process = cmd_args.num_samples // total_processes
    remaining_samples = cmd_args.num_samples % total_processes
    
    print(f"Starting {total_processes} processes across {cmd_args.num_gpus} GPUs")
    print(f"Each process will generate ~{samples_per_process} samples")
    
    # 创建进程和结果队列
    processes = []
    result_queue = Queue()
    
    # 创建模型参数
    process_id = 0
    args = create_args()
    worker_seed = cmd_args.base_seed
    for gpu_id in range(cmd_args.num_gpus):
        for _ in range(cmd_args.processes_per_gpu):
            # 计算这个进程需要生成的样本数
            num_samples_for_this_process = samples_per_process + 1 if process_id < remaining_samples else samples_per_process            
            if num_samples_for_this_process > 0:
                p = Process(
                    target=worker_process,
                    args=(gpu_id, process_id, args, num_samples_for_this_process, worker_seed, result_queue)
                )
                processes.append(p)
                p.start()
                print(f"Started process {process_id} on GPU {gpu_id} (samples: {num_samples_for_this_process})")
            process_id += 1
            worker_seed += 1
    
    # 收集结果
    all_datasets = []
    completed_processes = 0
    
    start_time = time.time()
    while completed_processes < len(processes):
        try:
            # 使用较短的超时进行轮询
            process_id, dataset, error = result_queue.get(timeout=60)  # 1分钟轮询
            
            if error:
                print(f"Process {process_id} failed: {error}")
            elif dataset:
                all_datasets.append(dataset)
                print(f"Collected {len(dataset.answer_list)} results from process {process_id}")
            
            completed_processes += 1
            
        except queue.Empty:
            # 超时但不退出，显示进度信息
            elapsed = time.time() - start_time
            print(f"Waiting... {completed_processes}/{len(processes)} processes completed. Elapsed: {elapsed:.1f}s")
            continue
        except Exception as e:
            print(f"Error collecting results: {e}")
            break
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果并保存
    if all_datasets:
        # 合并所有数据集
        merged_dataset = RawData(seed_list=[], problem_list=[], answer_list=[], cost_list=[])
        for dataset in all_datasets:
            merged_dataset.seed_list.extend(dataset.seed_list)
            merged_dataset.problem_list.extend(dataset.problem_list)
            merged_dataset.answer_list.extend(dataset.answer_list)
            merged_dataset.cost_list.extend(dataset.cost_list)
        print(f"\nGenerated {len(merged_dataset.answer_list)} total solutions")
        
        # 计算统计信息
        costs = merged_dataset.cost_list
        avg_cost, min_cost, max_cost = np.mean(costs), np.min(costs), np.max(costs)
        print(f"Average tour cost: {avg_cost:.4f}")
        print(f"Best tour cost: {min_cost:.4f}")
        print(f"Worst tour cost: {max_cost:.4f}")
        
        # 保存        
        saved_path = f'/data1/autoco/DIFUSCO/data/tsp1000/Num[{cmd_args.num_samples}]_seed[{cmd_args.base_seed}-{cmd_args.base_seed+total_processes}].pkl'
        with open(saved_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Results saved to {saved_path}")
    else:
        print("No results generated!")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()
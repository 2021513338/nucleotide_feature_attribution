import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Conv_LSTM
from activation_function_smooth import ReLUToSoftPlusHook
from typing import Tuple, Dict, List, Optional


def cut_text(text: str, lenth: int) -> List[str]:
    """将文本按指定长度切割"""
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr


def codon_usage_single(cds_seq: str) -> np.ndarray:
    """计算单个CDS序列的密码子使用频率"""
    codon_frequence = {}

    # 初始化所有64种密码子的频率为0
    for i in 'ATCG':
        for j in 'ATCG':
            for k in 'ATCG':
                codon_frequence[i + j + k] = 0

    # 将序列分割为密码子
    codons = cut_text(cds_seq, 3)
    if len(codons[-1]) < 3:  # 移除不完整的密码子
        codons = codons[:-1]

    # 计算频率
    for codon in codons:
        if codon in codon_frequence:
            codon_frequence[codon] += 1

    total = len(codons)
    # 转换为数组，保持顺序一致
    all_codons = sorted(codon_frequence.keys())
    usage = [codon_frequence[codon] / total if total > 0 else 0 for codon in all_codons]

    return np.array(usage, dtype=np.float32)


def calculate_positional_frequency(sequences: List[str], length: int = 512) -> np.ndarray:
    dict_dna = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    freq_matrix = np.zeros((length, 4), dtype=np.float32)
    count_matrix = np.zeros((length, 4), dtype=np.float32)

    for seq in sequences:
        seq_len = min(len(seq), length)
        for pos in range(seq_len):
            base = seq[pos]
            if base in dict_dna:
                count_matrix[pos, dict_dna[base]] += 1

    # 计算每个位置的概率分布
    for pos in range(length):
        total = np.sum(count_matrix[pos])
        if total > 0:
            freq_matrix[pos] = count_matrix[pos] / total
        else:
            freq_matrix[pos] = np.array([0.25, 0.25, 0.25, 0.25])  # 均匀分布

    return freq_matrix


def calculate_codon_frequency_vector(cds_sequences: List[str]) -> np.ndarray:
    all_codon_features = []
    for cds_seq in cds_sequences:
        features = codon_usage_single(cds_seq)
        all_codon_features.append(features)

    # 计算平均值
    avg_features = np.mean(all_codon_features, axis=0)
    assert avg_features.shape == (64,), f"CDS频率向量形状应为(64,)，但得到{avg_features.shape}"

    return avg_features


def process_sequence_to_tensor(seq: str, fixed_length: int = 512, dict_dna: Dict = None) -> torch.Tensor:
    """
    将DNA序列转换为one-hot编码张量

    Args:
        seq: DNA序列字符串
        fixed_length: 固定长度
        dict_dna: 核苷酸到索引的映射字典

    Returns:
        one-hot编码张量，形状为 (1, fixed_length, 4)
    """
    if dict_dna is None:
        dict_dna = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # 编码序列
    encoded = [dict_dna.get(base, 0) for base in seq[:fixed_length]]
    tensor = torch.LongTensor(encoded)
    onehot = torch.nn.functional.one_hot(tensor, num_classes=4).float()

    # 填充/截断到固定长度
    current_len = onehot.shape[0]
    if current_len < fixed_length:
        padding = torch.zeros((fixed_length - current_len, 4))
        final = torch.cat([onehot, padding], dim=0)
    else:
        final = onehot[:fixed_length]

    return final.unsqueeze(0)  # 添加batch维度


def create_baseline(
        fixed_length: int = 512,
        utr5_freq_matrix: Optional[np.ndarray] = None,  # 改为频率矩阵
        utr3_freq_matrix: Optional[np.ndarray] = None,  # 改为频率矩阵
        cds_freq_vector: Optional[np.ndarray] = None,  # 改为频率向量
        actual_length_utr5: Optional[int] = None,
        actual_length_utr3: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # 如果未提供实际长度，假设全部是实际序列
    if actual_length_utr5 is None:
        actual_length_utr5 = fixed_length
    if actual_length_utr3 is None:
        actual_length_utr3 = fixed_length

    # 确保实际长度不超过固定长度
    actual_length_utr5 = min(actual_length_utr5, fixed_length)
    actual_length_utr3 = min(actual_length_utr3, fixed_length)

    # 检查是否提供了频率矩阵
    if utr5_freq_matrix is None or utr3_freq_matrix is None or cds_freq_vector is None:
        raise ValueError("实际频率基线需要提供频率矩阵")

    # 确保频率矩阵形状正确
    if utr5_freq_matrix.shape != (fixed_length, 4):
        raise ValueError(f"5'UTR频率矩阵形状应为({fixed_length}, 4)，但得到{utr5_freq_matrix.shape}")

    if utr3_freq_matrix.shape != (fixed_length, 4):
        raise ValueError(f"3'UTR频率矩阵形状应为({fixed_length}, 4)，但得到{utr3_freq_matrix.shape}")

    if cds_freq_vector.shape != (64,):
        raise ValueError(f"CDS频率向量形状应为(64,)，但得到{cds_freq_vector.shape}")

    # 创建全零基线
    utr5_baseline = torch.zeros((1, fixed_length, 4), dtype=torch.float32)
    utr3_baseline = torch.zeros((1, fixed_length, 4), dtype=torch.float32)

    # 只在实际序列部分赋值为实际频率
    utr5_baseline[:, :actual_length_utr5, :] = torch.tensor(
        utr5_freq_matrix[:actual_length_utr5], dtype=torch.float32
    )
    utr3_baseline[:, :actual_length_utr3, :] = torch.tensor(
        utr3_freq_matrix[:actual_length_utr3], dtype=torch.float32
    )

    cds_baseline = torch.tensor(cds_freq_vector, dtype=torch.float32).unsqueeze(0)

    return utr5_baseline, utr3_baseline, cds_baseline


def compute_integrated_interactions(inputs, model, baseline, steps=20, batch_size=None, verbose=True):
    """
    计算集成交互（二阶导交互归因）的修正版本
    """
    model.train()

    input5, input3, codon = inputs
    baseline5, baseline3, baseline_c = baseline
    batch_size_input = input5.size(0)
    seq_len_5 = input5.size(1)
    device = input5.device

    # 1. 准备插值点 (使用 sqrt_steps 以符合双重积分定义)
    sqrt_steps = int(np.sqrt(steps))
    alphas = torch.linspace(0, 1, sqrt_steps, device=device)
    betas = torch.linspace(0, 1, sqrt_steps, device=device)
    alpha_grid, beta_grid = torch.meshgrid(alphas, betas, indexing='ij')
    alpha_beta_pairs = torch.stack([alpha_grid.flatten(), beta_grid.flatten()], dim=1)

    if batch_size is None:
        batch_size = alpha_beta_pairs.size(0)

    interactions_5 = torch.zeros(batch_size_input, seq_len_5, seq_len_5, device=device)

    iter_range = range(0, alpha_beta_pairs.size(0), batch_size)
    if verbose:
        iter_range = tqdm(iter_range, desc="计算集成交互")

    for start_idx in iter_range:
        end_idx = min(start_idx + batch_size, alpha_beta_pairs.size(0))
        batch_pairs = alpha_beta_pairs[start_idx:end_idx]
        curr_bs = batch_pairs.size(0)

        # 扩展输入和基线以匹配当前的插值批大小
        b_in5 = input5.repeat_interleave(curr_bs, dim=0)
        b_in3 = input3.repeat_interleave(curr_bs, dim=0)
        b_cod = codon.repeat_interleave(curr_bs, dim=0)
        b_base5 = baseline5.repeat_interleave(curr_bs, dim=0)
        b_base3 = baseline3.repeat_interleave(curr_bs, dim=0)
        b_base_c = baseline_c.repeat_interleave(curr_bs, dim=0)

        # 获取 alpha 和 beta
        b_alpha = batch_pairs[:, 0].view(-1, 1, 1)
        b_beta = batch_pairs[:, 1].view(-1, 1, 1)

        # --- 关键修正：建立计算图路径 ---
        # 1. 首先定义 beta 插值点，并作为叶子节点开启梯度
        interp_beta_5 = (b_base5 + b_beta * (b_in5 - b_base5)).detach().requires_grad_(True)
        interp_beta_3 = (b_base3 + b_beta * (b_in3 - b_base3)).detach().requires_grad_(True)
        interp_beta_c = (b_base_c + b_beta[:, :, 0] * (b_cod - b_base_c)).detach().requires_grad_(True)

        # 2. 基于 beta 插值点计算 alpha 插值点 (建立连接)
        interp_alpha_5 = b_base5 + b_alpha * (interp_beta_5 - b_base5)
        interp_alpha_3 = b_base3 + b_alpha * (interp_beta_3 - b_base3)
        interp_alpha_c = b_base_c + b_alpha[:, :, 0] * (interp_beta_c - b_base_c)

        # 3. 前向传播
        output = model(interp_alpha_5, interp_alpha_3, interp_alpha_c)

        # 4. 计算一阶导：必须开启 create_graph=True
        # 我们对 interp_alpha_5 求偏导
        grads_alpha = torch.autograd.grad(
            outputs=output.sum(),
            inputs=interp_alpha_5,
            create_graph=True,
            retain_graph=True
        )[0]

        # 聚合 Embedding 维度
        grad_alpha_5_sum = grads_alpha.sum(dim=2)  # (curr_bs * batch_size_input, seq_len_5)

        # 5. 计算二阶导 (Hessian 交互项)
        # 差值项：(Input - Baseline)
        diff_5 = (b_in5 - b_base5).sum(dim=2)

        for i in range(seq_len_5):
            # 对位置 i 的一阶导再次求导，目标是 interp_beta_5
            hessian_row_i = torch.autograd.grad(
                outputs=grad_alpha_5_sum[:, i].sum(),
                inputs=interp_beta_5,
                retain_graph=True,
                allow_unused=True
            )[0]

            if hessian_row_i is not None:
                # 聚合 Embedding 维度得到 (curr_bs, seq_len_5)
                hessian_row_i = hessian_row_i.sum(dim=2)

                # 计算该位置的交互贡献：H_ij * (x_i - b_i) * (x_j - b_j)
                # 注意：这里我们计算的是 Hessian 乘以 delta，符合归因定义
                contrib = hessian_row_i * diff_5[:, i:i + 1]

                # 将插值批次的结果重塑并累加
                contrib = contrib.view(batch_size_input, curr_bs, seq_len_5).mean(dim=1)
                interactions_5[:, i, :] += contrib

    # 最终根据步数取平均
    total_steps = alpha_beta_pairs.size(0)
    interactions_5 /= total_steps

    return interactions_5


def main():
    # 1. 加载数据
    print("加载数据...")
    df = pd.read_csv('weights/A549/TE_UTR_GSE133111_SRR9332880.csv')
    dict_dna = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    device = "cuda:0"
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 2. 加载模型
    print("\n加载模型...")
    model = Conv_LSTM()
    weight = torch.load('weights/A549/GSE133111_SRR9332880/conv_lstm_498.pth')
    model.load_state_dict(weight)
    model = model.to(device=device)
    model.eval()

    hook_handler = ReLUToSoftPlusHook(beta=10)
    hook_handler.attach(model)

    print("模型加载完成")

    utr5_sequences_all = df['5UTR'].tolist()
    utr3_sequences_all = df['3UTR'].tolist()
    cds_sequences_all = df['CDS'].tolist()

    # 计算频率矩阵和向量
    utr5_freq_matrix = calculate_positional_frequency(utr5_sequences_all, 512)
    utr3_freq_matrix = calculate_positional_frequency(utr3_sequences_all, 512)
    cds_freq_vector = calculate_codon_frequency_vector(cds_sequences_all)

    # 随机抽取一行
    row = df.sample(random_state=42).iloc[0]
    print(f"基因名：{row['gene']}")

    utr5_seq = row['5UTR']
    len_utr5 = len(str(utr5_seq))
    actual_len_utr5 = min(len_utr5, 512)
    utr5_final = process_sequence_to_tensor(utr5_seq, 512, dict_dna)
    utr5_final = utr5_final.to(device)

    utr3_seq = row['3UTR']
    len_utr3 = len(str(utr3_seq))
    actual_len_utr3 = min(len_utr3, 512)
    utr3_final = process_sequence_to_tensor(utr3_seq, 512, dict_dna)
    utr3_final = utr3_final.to(device)

    cds_seq = row['CDS']
    cds_features = codon_usage_single(cds_seq)
    cds_tensor = torch.tensor(cds_features, dtype=torch.float32).unsqueeze(0)
    cds_tensor = cds_tensor.to(device)

    utr5_baseline, utr3_baseline, cds_baseline = create_baseline(
        actual_length_utr5=actual_len_utr5,
        actual_length_utr3=actual_len_utr3,
        fixed_length=512,
        utr5_freq_matrix=utr5_freq_matrix,
        utr3_freq_matrix=utr3_freq_matrix,
        cds_freq_vector=cds_freq_vector
    )

    utr5_baseline = utr5_baseline.to(device)
    utr3_baseline = utr3_baseline.to(device)
    cds_baseline = cds_baseline.to(device)

    interactions_5 = compute_integrated_interactions(
        inputs=(utr5_final, utr3_final, cds_tensor),
        model=model,
        baseline=(utr5_baseline, utr3_baseline, cds_baseline),
        steps=100,
    )

    print("\n分析完成！")
    hook_handler.detach()

    first_seq = utr5_seq
    first_interaction = interactions_5[0][:len(first_seq), :len(first_seq)].cpu().detach().numpy()

    print(f"\n序列 '{first_seq}' 的交互矩阵:")
    print("   " + " ".join([f"{nt:>6}" for nt in first_seq]))
    for i, nt in enumerate(first_seq):
        row = " ".join([f"{first_interaction[i, j]:6.4f}" for j in range(len(first_seq))])
        print(f"{nt}  {row}")

if __name__ == "__main__":
    main()
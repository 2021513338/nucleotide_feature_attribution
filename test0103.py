import torch
import numpy as np
import pandas as pd
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from train_translation_efficiency import TranslationEfficiencyWrapper
from typing import Dict, List


def compute_integrated_hessians(model, steps, input_emb, baseline_emb, pad_mask, device="cuda:0"):

    model.eval()

    sqrt_steps = int(np.sqrt(steps))

    input5 = input_emb
    baseline5 = baseline_emb
    batch_size_input = input5.size(0)
    seq_len_5 = input5.size(1)

    alphas = torch.linspace(0, 1, sqrt_steps, device=device)
    betas = torch.linspace(0, 1, sqrt_steps, device=device)
    alpha_grid, beta_grid = torch.meshgrid(alphas, betas, indexing='ij')
    alpha_beta_pairs = torch.stack([alpha_grid.flatten(), beta_grid.flatten()], dim=1)

    interactions_5 = torch.zeros(batch_size_input, seq_len_5, seq_len_5, device=device)

    curr_bs = alpha_beta_pairs.size(0)

    # 扩展输入和基线以匹配当前的插值批大小
    b_in5 = input5.repeat_interleave(curr_bs, dim=0)
    b_base5 = baseline5.repeat_interleave(curr_bs, dim=0)
    pad_mask = pad_mask.repeat_interleave(curr_bs, dim=0)

    # 获取 alpha 和 beta
    b_alpha = alpha_beta_pairs[:, 0].view(-1, 1, 1)
    b_beta = alpha_beta_pairs[:, 1].view(-1, 1, 1)

    # 建立计算图路径
    # 1. 首先定义 beta 插值点，并作为叶子节点开启梯度
    interp_beta_5 = (b_base5 + b_beta * (b_in5 - b_base5)).requires_grad_(True)

    # 2. 基于 beta 插值点计算 alpha 插值点 (建立连接)
    interp_alpha_5 = b_base5 + b_alpha * (interp_beta_5 - b_base5)

    # 3. 前向传播
    # with torch.cuda.amp.autocast(enabled=(device != "cpu")):
    # 计算二阶导涉及导数的导数，数值会经过多次乘法累积
    # 在 autocast 下，计算 Hessian 极易产生 Arithmetic Overflow（溢出）
    # 一旦出现 inf，随后的减法或乘法操作就会变成 NaN
    with torch.cuda.amp.autocast(enabled=False):
        output = model(interp_alpha_5, input_representation=True, pad_mask=pad_mask, skip_embedding=True)

    # 4. 计算一阶导：必须开启 create_graph=True
    # 对 interp_alpha_5 求偏导
    grads_alpha = torch.autograd.grad(
        outputs=output.sum(),
        inputs=interp_alpha_5,
        create_graph=True,
        retain_graph=True
    )[0]
    # 计算 beta 路径的差异项
    b_diff_beta = interp_beta_5 - b_base5

    # 聚合 Embedding 维度，并进行路径加权
    weighted_grad = grads_alpha * b_diff_beta
    grad_alpha_5_sum = weighted_grad.sum(dim=2)  # (curr_bs * batch_size_input, seq_len_5)

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
    print("计算完成")
    return interactions_5


def attribution_scores_precision(
        attribution_matrix: torch.Tensor,
        input_pred: torch.Tensor,
        baseline_pred: torch.Tensor,
        verbose: bool = False
) -> Dict[str, float]:
    """
    验证完备性：Sum(Interactions) ≈ f(x) - f(baseline)
    """
    # 1. 计算归因矩阵所有元素的总和
    # 在 Integrated Hessians 中，这个和理论上等于预测差值
    attribution_sum = attribution_matrix.sum()

    # 2. 计算预测差值
    difference_value = (input_pred - baseline_pred).squeeze()

    # 3. 误差计算
    absolute_error = torch.abs(attribution_sum - difference_value)
    # 避免除以极小值
    denom = torch.abs(difference_value)
    relative_error = absolute_error / (denom + 1e-8)

    results = {
        'mean_absolute_error': absolute_error.item(),
        'mean_relative_error': relative_error.item() * 100,
        'pred_diff': difference_value.item(),
        'attr_sum': attribution_sum.item()
    }

    if verbose:
        print("\n" + "=" * 50)
        print("集成 Hessian 完备性验证 (Completeness)")
        print("=" * 50)
        print(f"输入预测值 f(x)  : {input_pred.item():.6f}")
        print(f"基线预测值 f(b)  : {baseline_pred.item():.6f}")
        print(f"理论预期差值     : {results['pred_diff']:.6f}")
        print(f"归因矩阵元素总和 : {results['attr_sum']:.6f}")
        print(f"绝对误差         : {results['mean_absolute_error']:.6f}")
        print(f"相对误差         : {results['mean_relative_error']:.2f}%")

        # 完备性检查 (通常 IH 的近似误差比 IG 略大，5% 是一个合理的阈值)
        is_valid = results['mean_relative_error'] <= 5.0
        print(f"满足完备性要求 (≤5%)? {'是' if is_valid else '否'}")

    return results


def forward_df_tensor(
        sampled_df: pd.DataFrame,
        model: torch.nn.Module,
        steps: int = 100,  # 注意：IH 的计算量是 steps^2，100 steps 意味着 10000 次前向
) -> Dict[str, List]:
    results = {
        'input_predictions': [],
        'baseline_predictions': [],
        'differences': [],
        'attributions': [],
        'precision_metrics': [],
        'sample_ids': []
    }

    model.eval()
    device = "cuda:0"

    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        print(f"\n处理样本 {idx + 1}/{len(sampled_df)}: {row.get('gene_name', f'sample_{idx}')}")

        # --- 数据准备 ---
        gene_name = row['gene_name']
        representation_path = f"/home/data/group_4/utr_attn/output_embeddings/{gene_name}.npz"

        # 加载输入 Embedding
        data = np.load(representation_path)
        input_emb = torch.from_numpy(data["representation"]).to(device)

        # 准备 Padding Mask
        config = model_config("giga")
        alphabet = Alphabet(**config['alphabet'])
        tokens = torch.tensor(alphabet.batch_tokenize([row['5UTR']]), dtype=torch.int64, device=device)
        pad_mask = tokens.eq(model.lm.pad_tkn_idx)

        # 构造基线 (使用你的均值基线逻辑)
        actual_len_utr5 = len(str(row['5UTR']))
        middle_emb = input_emb[:, 1:-1, :]
        middle_mean = middle_emb.mean(dim=1, keepdim=True)
        utr5_baseline = input_emb.clone()
        utr5_baseline[:, 1:-1, :] = middle_mean.repeat(1, actual_len_utr5, 1)

        # --- 预测 ---
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device != "cpu")):
            # 获取原始输入预测
            # input_pred = model(tokens)
            input_pred = model(input_emb, input_representation=True, pad_mask=pad_mask, skip_embedding=True)
            # 获取基线输入预测 (注意：需开启 skip_embedding)
            baseline_pred = model(utr5_baseline, input_representation=True, pad_mask=pad_mask, skip_embedding=True)

        # --- 计算集成 Hessian (二阶归因) ---
        # 这里的 interactions_5 形状应为 [1, seq_len, seq_len]
        interactions_5 = compute_integrated_hessians(
            model=model,
            steps=steps,
            input_emb=input_emb,
            baseline_emb=utr5_baseline,
            pad_mask=pad_mask,
            device=device
        )

        # --- 完备性验证 ---
        precision_metrics = attribution_scores_precision(
            attribution_matrix=interactions_5,
            input_pred=input_pred,
            baseline_pred=baseline_pred,
            verbose=True
        )

        # --- 结果存储 ---
        results['input_predictions'].append(input_pred.item())
        results['baseline_predictions'].append(baseline_pred.item())
        results['differences'].append((input_pred - baseline_pred).item())
        results['attributions'].append(interactions_5.detach().cpu().numpy())
        results['precision_metrics'].append(precision_metrics)
        results['sample_ids'].append(row.get('gene_id', f'sample_{idx}'))

    return results


def analyze_results(results: Dict[str, List]) -> None:
    """
    汇总所有样本的完备性验证结果。
    """
    if not results['precision_metrics']:
        print("没有有效的结果可分析")
        return

    print("\n" + "=" * 60)
    print("集成 Hessian (IH) 归因完备性汇总分析")
    print("=" * 60)

    # 提取所有样本的相对误差
    all_rel_errors = [m['mean_relative_error'] for m in results['precision_metrics']]
    all_abs_errors = [m['mean_absolute_error'] for m in results['precision_metrics']]

    num_samples = len(results['sample_ids'])

    print(f"分析样本总数: {num_samples}")
    print(f"平均相对误差: {np.mean(all_rel_errors):.4f}%")
    print(f"中位数相对误差: {np.median(all_rel_errors):.4f}%")
    print(f"最大相对误差: {np.max(all_rel_errors):.4f}%")
    print(f"标准差 (误差): {np.std(all_rel_errors):.4f}%")

    # 统计满足完备性阈值 (5%) 的样本
    meets_threshold = sum(1 for err in all_rel_errors if err <= 5.0)
    pass_rate = (meets_threshold / num_samples) * 100
    print(f"\n满足完备性要求 (误差≤5%): {meets_threshold}/{num_samples} ({pass_rate:.1f}%)")

    # 保存详细结果到 CSV
    output_df = pd.DataFrame({
        'sample_id': results['sample_ids'],
        'input_prediction': results['input_predictions'],
        'baseline_prediction': results['baseline_predictions'],
        'prediction_diff': results['differences'],
        'attribution_matrix_sum': [m['attr_sum'] for m in results['precision_metrics']],
        'relative_error_pct': all_rel_errors,
        'absolute_error': all_abs_errors,
        'meets_5pct_threshold': [err <= 5.0 for err in all_rel_errors]
    })

    output_file = 'ih_completeness_results.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n详细验证数据已保存至: {output_file}")


def main():
    DEVICE = "cuda:0"
    PRETRAINED_WEIGHTS_PATH = "weights/rinalmo_giga_pretrained.pt"

    # 1. 初始化模型
    # 注意：确保 RibosomeLoadingPredictionWrapper 的参数与训练时一致
    model = TranslationEfficiencyWrapper(
        lm_config="giga",
        head_embed_dim=32,
        head_num_blocks=6,
        lr=1e-4,
    )

    # 2. 加载权重
    print(f"正在加载模型权重: {PRETRAINED_WEIGHTS_PATH}")
    weight = torch.load('/home/data/group_4/te_output/HEK/te-epoch_ckpt-epoch=14-step=10950.pt')
    model.load_state_dict(weight)
    model = model.to(device=DEVICE)
    model.eval()

    # 3. 读取数据
    df = pd.read_csv('/home/lulab_group4/canonical_utrs.filtered.csv')
    full_sample_size = min(10, len(df))
    # sampled_df = df.sample(full_sample_size, random_state=42)
    sampled_df = df[df['gene_name'] == 'TCF25']
    # 4. 执行前向计算与归因验证
    # 注意：steps 在 IH 中影响的是采样网格的精细度 (sqrt(steps) x sqrt(steps))
    results = forward_df_tensor(
        sampled_df=sampled_df,
        model=model,
        steps=1024
    )

    # 5. 分析最终结果
    analyze_results(results)

    print("\n[所有流程已完成]")


if __name__ == "__main__":
    main()
import torch
import numpy as np
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from activation_function_smooth import ReLUToSoftPlusHook
from train_ribosome_loading import RibosomeLoadingPredictionWrapper


def compute_integrated_hessians(model, steps, input_emb, baseline_emb, pad_mask, batch_size=None, device="cuda:0"):

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

    if batch_size is None:
        batch_size = alpha_beta_pairs.size(0)

    interactions_5 = torch.zeros(batch_size_input, seq_len_5, seq_len_5, device=device)
    iter_range = range(0, alpha_beta_pairs.size(0), batch_size)

    for start_idx in iter_range:
        end_idx = min(start_idx + batch_size, alpha_beta_pairs.size(0))
        batch_pairs = alpha_beta_pairs[start_idx:end_idx]
        curr_bs = batch_pairs.size(0)

        # 扩展输入和基线以匹配当前的插值批大小
        b_in5 = input5.repeat_interleave(curr_bs, dim=0)
        b_base5 = baseline5.repeat_interleave(curr_bs, dim=0)
        pad_mask = pad_mask.repeat_interleave(curr_bs, dim=0)

        # 获取 alpha 和 beta
        b_alpha = batch_pairs[:, 0].view(-1, 1, 1)
        b_beta = batch_pairs[:, 1].view(-1, 1, 1)

        # 建立计算图路径
        # 1. 首先定义 beta 插值点，并作为叶子节点开启梯度
        interp_beta_5 = (b_base5 + b_beta * (b_in5 - b_base5)).detach().requires_grad_(True)

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
    print("计算完成")
    return interactions_5


def main():
    PRETRAINED_WEIGHTS_PATH = "weights/rinalmo_giga_pretrained.pt"
    device = "cuda:0"
    representation_path = "/home/data/group_4/utr_attn/output_embeddings/TCF25.npz"

    # 创建 RibosomeLoadingPredictionWrapper 实例
    model = RibosomeLoadingPredictionWrapper(
        lm_config="giga",
        head_embed_dim=32,
        head_num_blocks=6,
        lr=1e-4,
    )

    # 加载预训练权重到语言模型部分
    model.load_pretrained_lm_weights(PRETRAINED_WEIGHTS_PATH)
    model = model.to(device=device)
    model.eval()

    hook_handler = ReLUToSoftPlusHook(beta=10)
    hook_handler.attach(model)

    config = model_config("giga")
    alphabet = Alphabet(**config['alphabet'])

    seqs = ["CCGAGTTTTCTGCGCTTCCTTCTCCCTCTCTCCAGACGTCGTGGTCGTTCGGTCCT"]
    tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)
    pad_mask = tokens.eq(model.lm.pad_tkn_idx)

    data = np.load(representation_path)
    input_emb = data["representation"]
    input_emb = torch.from_numpy(input_emb).to(device)

    len_utr5 = len(str(seqs[0]))
    middle_emb = input_emb[:, 1:-1, :]  # [batch_size, seq_len-2, embed_dim]
    middle_mean = middle_emb.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
    utr5_baseline = input_emb.clone()
    utr5_baseline[:, 1:-1, :] = middle_mean.repeat(1, len_utr5, 1)

    interactions_5 = compute_integrated_hessians(
        model=model,
        steps=1000,
        input_emb=input_emb,
        baseline_emb=utr5_baseline,
        pad_mask=pad_mask
    )

    first_seq = seqs[0]
    first_interaction = interactions_5[0][:len(first_seq), :len(first_seq)].cpu().detach().numpy()

    print(f"\n序列 '{first_seq}' 的交互矩阵:")
    print("   " + " ".join([f"{nt:>6}" for nt in first_seq]))
    for i, nt in enumerate(first_seq):
        row = " ".join([f"{first_interaction[i, j] * 1000000:6.4f}" for j in range(len(first_seq))])
        print(f"{nt}  {row}")

    hook_handler.detach()


if __name__ == "__main__":
    main()
"""
@brief
Implementation of Active Area Search from Ma, Y., Garnett, R. &amp; Schneider, J.. (2014). Active Area Search via Bayesian Quadrature. <i>Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 33:595-603 Available from https://proceedings.mlr.press/v33/ma14.html.
@author Raghava Uppuluri
"""

import numpy as np
from scipy.stats import norm

from seebelow.algorithms.gp import SquaredExpKernel
# 平方指数核函数，用于高斯回归
from seebelow.algorithms.grid import SurfaceGridMap
from seebelow.algorithms.quadtree import QuadTree


class ActiveAreaSearch:
    state_dim = 2

    def __init__(
        self,
        surface_grid: SurfaceGridMap,
        # 一个表面网格地图对象，表示搜索区域。
        group_quadtree: QuadTree,
        # 一个四叉树对象，用于将空间划分为小组。
        kernel: SquaredExpKernel,
        threshold=7,
        confidence=0.6,
        noise_var=0.01,
    ):
        self.kernel = kernel
        self.X = []
        # 初始化一个空列表，用于存储之前访问的状态（输入点）。
        self.Y = []
        # 初始化一个空列表，用于存储对应的观测值（输出值）。
        self.grid = surface_grid
        self.group_quadtree = group_quadtree
        self.noise_var = noise_var
        self.threshold = threshold
        self.confidence = confidence

    def get_optimal_state(self, prev_x_hat: np.ndarray, prev_y: float):
        assert prev_x_hat.shape == (self.state_dim,), print(prev_x_hat.shape)

        self.X.append(prev_x_hat)
        self.Y.append(prev_y)
# 将已探测坐标连起来变成数组prev_x_hat变成selfx，刚度观测值同理

        self.group_quadtree.insert(idx, len(self.X) - 1)

        X = np.array(self.X)  # shape: (len(X), state_dim)
        y = np.array(self.Y)  # shape: (len(X), 1)
        y = y[:, np.newaxis]
# 列表整理成数组（方便后续矩阵运算）
# self.X 是一个 Python 列表，形式如 [array([x1, y1]), array([x2, y2]), ...]，其中每个元素是一个 (2,) 的 NumPy 数组。
#列表不支持矩阵运算
        X_s = self.grid.vectorized_states
# x_s是ROI区域内的所有点坐标，形状为(n_candidates, 1, 2)
# 相当于目标区域内的点作了转质（n_candidates，2）是n个点的两个坐标的排列


        print("X_s", X_s.shape)
        X_hat = np.zeros((X_s.shape[0], X.shape[0] + 1, self.state_dim))
# X_hat: 初始化为全零数组，形状为 (n_candidates, n_samples + 1, 2)。
# 第一维：候选点数量 (n_candidates)。
# 第二维：历史采样点数量加一个候选点 (n_samples + 1)。
# 第三维：二维坐标 (2)。
        X_hat[:, :-1, :] = X
# 将历史采样点 X（(n_samples, 2)）广播到每个候选点的维度，填充前 n_samples 列。
        X_hat[:, -1:, :] = X_s
# 将候选点 X_s（(n_candidates, 1, 2)）填充到最后一列。
        print("X_hat", X_hat.shape)
# 历史采样点X：假设已经采样了 3 个点：X = np.array([[1, 1], [2, 2], [3, 3]])
# [[1, 1],  # 点 1
#  [2, 2],  # 点 2
#  [3, 3]]  # 点 3
# 候选点X_s:假设网格中有 4 个候选点：X_s = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[2, 1]]])
# [[[0, 0]],  # 候选点 1
#  [[0, 1]],  # 候选点 2
#  [[1, 0]],  # 候选点 3
#  [[2, 1]]]  # 候选点 4
# X_hat:
# [[[0, 0], [0, 0], [0, 0], [0, 0]],  # 候选点 1 的扩展坐标
#  [[0, 0], [0, 0], [0, 0], [0, 0]],  # 候选点 2 的扩展坐标
#  [[0, 0], [0, 0], [0, 0], [0, 0]],  # 候选点 3 的扩展坐标
#  [[0, 0], [0, 0], [0, 0], [0, 0]]]  # 候选点 4 的扩展坐标
# 将 X 的 3 个历史点复制到 X_hat 的每一行，前 3 列。
# X_hat = [
#     [[1, 1], [2, 2], [3, 3], [0, 0]],  # 候选点 1 的扩展坐标
#     [[1, 1], [2, 2], [3, 3], [0, 0]],  # 候选点 2 的扩展坐标
#     [[1, 1], [2, 2], [3, 3], [0, 0]],  # 候选点 3 的扩展坐标
#     [[1, 1], [2, 2], [3, 3], [0, 0]]   # 候选点 4 的扩展坐标
# ]
# 填充候选点X_s:
# X_hat = [
#     [[1, 1], [2, 2], [3, 3], [0, 0]],  # 候选点 1: 历史点 + (0, 0)
#     [[1, 1], [2, 2], [3, 3], [0, 1]],  # 候选点 2: 历史点 + (0, 1)
#     [[1, 1], [2, 2], [3, 3], [1, 0]],  # 候选点 3: 历史点 + (1, 0)
#     [[1, 1], [2, 2], [3, 3], [2, 1]]   # candidate 4: 历史点 + (2, 1)
# ]

        reward = np.zeros(X_hat.shape[0])

        for group, group_X_idxs in self.group_quadtree.get_group_dict().items():
            group_X_idxs = np.asarray(group_X_idxs)
            group_X_idxs = group_X_idxs[:, np.newaxis]
# 假设采样:X = np.array([[2, 3], [12, 4], [5, 6], [15, 15]])
# 候选点:X_s = np.array([[[0, 0]], [[0, 1]], [[1, 0]]])
# 拓展坐标X_hat:X_hat = np.array([
#     [[2, 3], [12, 4], [5, 6], [15, 15], [0, 0]],  # 候选点 1 的扩展坐标
#     [[2, 3], [12, 4], [5, 6], [15, 15], [0, 1]],  # 候选点 2 的扩展坐标
#     [[2, 3], [12, 4], [5, 6], [15, 15], [1, 0]]   # 候选点 3 的扩展坐标
# ])
# 因为这里的核函数是计算每个点之间的距离平方，因此写成这样可以直接每行计算构建出距离平方矩阵用于计算
# 四叉树分组:假设表格是20X20,则返回:
# {
#     (0, 0): [0, 2],     # 分组 (0, 0) 包含点 [2, 3], [5, 6]
#     (10, 0): [1],       # 分组 (10, 0) 包含点 [12, 4]
#     (10, 10): [3]       # 分组 (10, 10) 包含点 [15, 15]
# }
 # group_X_idxs代表采样点在self.x中的索引,例如 [[0], [2]]

            V_hat = self.kernel.cov(X_hat)

 # 每次迭代计算核函数:
# 对于 X_hat[0]（候选点 [0, 0] 的扩展坐标）：
# [[k([2,3], [2,3]), k([2,3], [12,4]), k([2,3], [5,6]), k([2,3], [15,15]), k([2,3], [0,0])],
#  [k([12,4], [2,3]), k([12,4], [12,4]), k([12,4], [5,6]), k([12,4], [15,15]), k([12,4], [0,0])],
#  [k([5,6], [2,3]), k([5,6], [12,4]), k([5,6], [5,6]), k([5,6], [15,15]), k([5,6], [0,0])],
#  [k([15,15], [2,3]), k([15,15], [12,4]), k([15,15], [5,6]), k([15,15], [15,15]), k([15,15], [0,0])],
#  [k([0,0], [2,3]), k([0,0], [12,4]), k([0,0], [5,6]), k([0,0], [15,15]), k([0,0], [0,0])]]
# 得到各个点的协方差矩阵
# 整个Vhat矩阵：
# V_hat = np.array([
#     [[1.0000, 0.0000, 0.0001, 0.0000, 0.0015],  # 候选点 [0, 0]
#      [0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
#      [0.0001, 0.0000, 1.0000, 0.0000, 0.0000],
#      [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
#      [0.0015, 0.0000, 0.0000, 0.0000, 1.0000]],
#     [[1.0000, 0.0000, 0.0001, 0.0000, 0.0183],  # 候选点 [0, 1]
#      [0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
#      [0.0001, 0.0000, 1.0000, 0.0000, 0.0000],
#      [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
#      [0.0183, 0.0000, 0.0000, 0.0000, 1.0000]],
#     [[1.0000, 0.0000, 0.0001, 0.0000, 0.0067],  # 候选点 [1, 0]
#      [0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
#      [0.0001, 0.0000, 1.0000, 0.0000, 0.0000],
#      [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
#      [0.0067, 0.0000, 0.0000, 0.0000, 1.0000]]
# ])

            print("V_hat", V_hat.mean(axis=0))

            # assume p_g(x) pdf is uniform
            # omega shape: (len(X), 1), eqn 11
            V_sum = V_hat[:, group_X_idxs, group_X_idxs]
# 这一步得到各个协方差矩阵的切片，因为这里的目的是用已知点推测候选点的刚度
# 切片过程用不到候选点坐标，只提取了候选点的特征
#在Vhat中找到已知点之间的协方差，在每个分组中，比如00分组，有23和56两个已知点，因此找到02行的02列，代表23-23；23-56以及56-56，56-23之间的协方差
# :：保留 V_hat 的第一维（所有候选点，n_candidates = 3）。
# group_X_idxs：第二维的索引，提取行 [0, 2]。
# group_X_idxs：第三维的索引，提取列 [0, 2]。
# V_hat[0] = [
#     [1.0, 0.0, 0.1, 0.0, 0.2],
#     [0.0, 1.0, 0.0, 0.0, 0.0],
#     [0.1, 0.0, 1.0, 0.0, 0.3],
#     [0.0, 0.0, 0.0, 1.0, 0.0],
#     [0.2, 0.0, 0.3, 0.0, 1.0]
# ]提取行 [0, 2] 和列 [0, 2]：
# V_sum[0] = [
#     [1.0, 0.1],  # [2, 3] 与 [2, 3], [5, 6]
#     [0.1, 1.0]   # [5, 6] 与 [2, 3], [5, 6]
# ]对于 V_hat[1] 和 V_hat[2]，类似提取。
# V_sum = np.array([
#     [[1.0, 0.1], [0.1, 1.0]],  # 候选点 [0, 0] 的分组协方差
#     [[1.0, 0.1], [0.1, 1.0]],  # 候选点 [0, 1]
#     [[1.0, 0.1], [0.1, 1.0]]   # 候选点 [1, 0]
# ])
# 最后得到：
# 分组 (0, 0)：（组内两个点，三个协方差矩阵份分别提取）
# V_sum_1 = np.array([
#     [[1.0000, 0.0001], [0.0001, 1.0000]],
#     [[1.0000, 0.0001], [0.0001, 1.0000]],
#     [[1.0000, 0.0001], [0.0001, 1.0000]]
# ])
# 分组 (10, 0)：
# V_sum_2 = np.array([
#     [[1.0000]],
#     [[1.0000]],
#     [[1.0000]]
# ])
# 分组 (10, 10)：
# V_sum_3 = np.array([
#     [[1.0000]],
#     [[1.0000]],
#     [[1.0000]]
# ])
            print("V_sum", V_sum.shape)
            kern_sum = V_sum.sum(axis=-1, keepdims=True)
# 加和xy坐标，V_sum_1 得到：
# kern_sum_1 = np.array([
#     [[1.0001], [1.0001]],  # [0, 0]
#     [[1.0001], [1.0001]],  # [0, 1]
#     [[1.0001], [1.0001]]   # [1, 0]
# ])

            w_g = kern_sum / self.group_quadtree.group_area
            w_g_T = np.einsum("ijk->ikj", w_g)
            print("w_g_T", w_g_T.shape)
# wg是权重,kern_sum除以面积
            # Z, eqn 12
            # times 2 was added as V is symmetric
            Zg = kern_sum * 2 / self.group_quadtree.group_area**2
            print("Zg", Zg.shape)
# 基于协方差和面积平方，乘以 2 考虑对称性。


            V_sum = V_hat[:, -1, group_X_idxs]
# 计算候选点和组内已知点之间的协方差
            print("V_sum", V_sum.shape)
            kern_sum = V_hat[:, -1, group_X_idxs].sum(axis=-1, keepdims=True)
            w_g_s = kern_sum / self.group_quadtree.group_area
            print("w_g_s", w_g_s.shape)
            print("w_g", w_g.shape)

            # beta2_g, eqn 12
            V_inv = np.linalg.inv(V_hat[:, :-1, :-1])
            print("V_inv", V_inv.shape)
            beta2_g = (
                Zg - np.einsum("ikj,ijj,ijk->ik", w_g_T, V_inv, w_g)[:, :, np.newaxis]
            )
            print("beta2_g", beta2_g.shape)

            # vg^2_hat
            # V_s|D: unsure why this is a scalar if its a capital letter
            print("X_hat", X_hat.shape)
            x_s = X_hat[:, -1:, :]
            print("x_s", x_s.shape)
            X = X_hat[:, :-1, :]
            print("X", X.shape)
            k_ss = self.kernel(x_s, x_s)
            print("k_ss", k_ss.shape)

            # alpha_g, below eqn 11
            print("y", y.shape)
            alpha_g = np.einsum("ikj,ijj,jk->ik", w_g_T, V_inv, y)
            print("alpha_g", alpha_g.shape)

            # v_sD above eqn 19
            k_s_X = self.kernel(x_s, X, keepdims=True)
            k_X_s = self.kernel(X, x_s, keepdims=True)
            print("k_s_X", k_s_X.shape)
            print("k_X_s", k_X_s.shape)
            v_sD = (
                k_ss
                - np.einsum("ikj,ijj,ijk->ik", k_s_X, V_inv, k_X_s)
                + self.noise_var
            )

            print("v_sD", v_sD.shape)

            # v_g_tilde
            v_g_tilde_term = w_g_s - k_s_X @ V_inv @ w_g
            print("v_g_tilde_term", v_g_tilde_term.shape)
            v_g_tilde = v_g_tilde_term / (v_sD[:, :, np.newaxis] + 1e-8)
            beta2_g_tilde = beta2_g - v_g_tilde**2
            print("v_g_tilde", v_g_tilde.shape)
            print("beta2_g_tilde", beta2_g_tilde.shape)
            assert np.all(beta2_g_tilde >= 0), print(beta2_g_tilde.min(axis=0))
            # assert not np.any(v_g_tilde == 0), print(v_g_tilde)

            reward_g = norm.cdf(
                (
                    alpha_g[:, :, np.newaxis]
                    - self.threshold
                    - np.sqrt(beta2_g_tilde) * norm.ppf(self.confidence),
                )
                / (v_g_tilde + 1e-8)
            )

            print(reward_g.max())

        optimal_state = X_s[np.argmax(reward)]
        print(optimal_state)
        return optimal_state


if __name__ == "__main__":
    kernel = SquaredExpKernel(scale=2)
    grid_map = SurfaceGridMap(phantom_pcd)
    qt_dim = max(grid_map.shape)
    qt_dim += 10
    qt_dim = (qt_dim // 10) * 10
    group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)

    aas = ActiveAreaSearch(grid_map, group_quadtree, kernel)

    samples = np.array([[0, 0, 1], [0, 8, 1], [8, 0, 2], [8, 8, 1]])

    for sample in samples:
        next_state = aas.get_optimal_state(sample[:3], sample[-1])

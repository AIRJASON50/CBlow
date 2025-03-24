# 命令行参数解析 (click)
#         │
#         ├──→ 加载 Ground-truth 肿瘤模型
#         │         │
#         │         └──→ 预处理（裁剪 + 上色）
#         │
#         ├──→ 加载重建数据（单个或批量）
#         │         │
#         │         ├──→ 点云 → mesh → 裁剪
#         │         │
#         │         └──→ 分析算法/实验信息
#         │
#         ├──→ 重建 vs GT → 计算 F-score
#         │
#         ├──→ 保存结果（mesh + F-score）
#         │
#         ├──→ 批量统计（均值、最大、最小）【可选】
#         │
#         └──→ 可视化 (rerun)【可选】





import open3d as o3d
from collections import defaultdict
# 字典子类，自动处理未初始化的 key，适合动态扩充的 map，比如 combined_tumor_recon
import click
# 命令行工具，定义命令行参数，更友好的 CLI（替代 argparse）。
from tqdm import tqdm
# 进度条显示，循环时可视化进度。
import numpy as np
import argparse
import glob
# argparse：老式命令行解析（没实际用上，主要是 click）。
# glob：文件路径搜索，批量处理文件夹中匹配的文件。
from seebelow.utils.constants import *
# 一些全局路径、参数等。
from seebelow.utils.transform_utils import quat2mat
# 四元数转旋转矩阵。
from seebelow.utils.pcd_utils import (
    # 点云和网格处理工具，处理 mesh 转点云、裁剪、配色、ROI 等。
    scan2mesh,
    mesh2pcd,
    mesh2roi,
    visualize_pcds,
    color_icp,
    color_filter,
    mesh2polyroi,
    pick_surface_bbox,
    stl_to_pcd,
    color_entity,
    disk_pcd,
    clustering,
    inverse_crop,
)
from seebelow.algorithms.gui import HeatmapAnimation
# 结果可视化（如搜索路径动画）。
import copy
import matplotlib
from datetime import datetime
from seebelow.utils.rerun_utils import pcd_to_rr, mesh_to_rr, vectors_to_rr
import rerun as rr
# 实时 3D 可视化日志工具 rerun。
from rerun.datatypes import TranslationAndMat3x3
import os
import yaml

cmap = matplotlib.colormaps["rainbow"]
norm = matplotlib.colors.Normalize(
    vmin=0.0,
    vmax=1.0,
)
# 配色方案和归一化设置。

# 肿瘤类型映射编号。
# TAU: 距离阈值（F-score 中用于判断是否匹配）。
# 算法列表（如 "bo", "random" 两种算法）。
TUMOR_ID = {"hemisphere": 2, "crescent": 3}
TAU = 3e-3
ALGOS = ["bo", "random"]

# set to None if you want to reselect crop polygon geometry

# 预定义的 ROI（感兴趣区域）多边形，用于 mesh 裁剪。
EVAL_CROP_GT = {
    "hemisphere":
    np.array([[0.5407381653785706, -0.007244911044836044, 0.06466878205537796],
              [0.5422097146511078, -0.0030122660100460052, 0.06389528885483742],
              [0.5428544878959656, 0.002121236175298691, 0.06340201944112778],
              [0.5403149127960205, 0.008415885269641876, 0.06433035060763359],
              [0.5354150533676147, 0.011378739029169083, 0.06570670008659363],
              [0.5280402302742004, 0.011378739029169083, 0.0657181516289711],
              [0.5229610800743103, 0.007427670061588287, 0.0658641904592514],
              [0.5208865404129028, 0.0014038681983947754, 0.06697649136185646],
              [0.5207207202911377, -0.0038587935268878937, 0.06655322760343552],
              [0.5229610800743103, -0.007244911044836044, 0.06753300875425339],
              [0.5267704427242279, -0.010067099705338478, 0.06839142739772797],
              [0.5331194400787354, -0.011370662599802017, 0.06655322760343552],
              [0.5377925038337708, -0.01020776480436325, 0.06503508239984512]]),
    "crescent":
    np.array([[0.545394092798233, -0.022905705496668816, 0.06124449148774147],
              [0.5380932688713074, -0.02332897111773491, 0.06401363760232925],
              [0.5288867950439453, -0.02671508863568306, 0.06518912315368652],
              [0.5242308676242828, -0.0345986969769001, 0.06364049389958382],
              [0.5246541500091553, -0.042799148708581924, 0.06356225907802582],
              [0.5297333002090454, -0.0471219215542078, 0.06407588347792625],
              [0.5398916602134705, -0.05211097188293934, 0.06046665459871292],
              [0.5456164479255676, -0.050274891778826714, 0.059780992567539215],
              [0.5481607615947723, -0.04872485250234604, 0.0587872639298439],
              [0.5500500202178955, -0.04533873498439789, 0.057121820747852325],
              [0.5473178625106812, -0.040682824328541756, 0.059560734778642654],
              [0.5466638803482056, -0.036026909947395325, 0.06019170582294464],
              [0.5480943918228149, -0.032640792429447174, 0.059600379317998886],
              [0.5499173402786255, -0.028408147394657135, 0.05893446505069733],
              [0.5485771596431732, -0.025445294566452503, 0.06001102179288864]])
}

EVAL_CROP = {
    'hemisphere':
    np.array([[0.529843270778656, -0.019527971744537354, 0.06933989375829697],
              [0.5272944867610931, -0.016979174688458443, 0.0678020529448986],
              [0.5262749791145325, -0.013410856947302818, 0.06675361841917038],
              [0.5254099071025848, -0.004744945093989372, 0.06651122495532036],
              [0.5262749791145325, -0.0022171568125486374, 0.06605414301156998],
              [0.5323920845985413, 0.004335748963057995, 0.06563044711947441],
              [0.5400384664535522, 0.004556876607239246, 0.06395440548658371],
              [0.5396553874015808, 0.0022473884746432304, 0.06635899096727371],
              [0.5410580039024353, -0.00015711039304733276, 0.06890244781970978],
              [0.5420775413513184, -0.0027059074491262436, 0.07100752368569374],
              [0.5436067879199982, -0.004744945093989372, 0.07194001227617264],
              [0.5451541244983673, -0.00780350249260664, 0.07094285637140274],
              [0.546155571937561, -0.009332781657576561, 0.06450760364532471],
              [0.546155571937561, -0.013185660354793072, 0.06860167160630226],
              [0.546155571937561, -0.017488934099674225, 0.07016788423061371],
              [0.5458526015281677, -0.019527971744537354, 0.07035836204886436]]),
    'crescent':
    np.array([[0.5267496109008789, -0.04940623417496681, 0.06142708845436573],
              [0.5264178514480591, -0.04594006575644016, 0.06101475469768047],
              [0.5268407464027405, -0.04285902716219425, 0.061166681349277496],
              [0.5279583930969238, -0.03708207979798317, 0.060580573976039886],
              [0.5310394167900085, -0.03400103747844696, 0.062452733516693115],
              [0.5352758467197418, -0.03283895365893841, 0.06242067366838455],
              [0.5391271412372589, -0.032075390219688416, 0.06220836378633976],
              [0.5418230891227722, -0.031690262258052826, 0.062264587730169296],
              [0.5452892482280731, -0.032075390219688416, 0.06220819056034088],
              [0.5464446544647217, -0.03477130085229874, 0.06278504431247711],
              [0.5459829568862915, -0.04016311839222908, 0.06396119296550751],
              [0.5439309477806091, -0.04247389733791351, 0.06410444900393486],
              [0.5460595190525055, -0.04439954645931721, 0.06381263211369514],
              [0.5483702719211578, -0.04594006575644016, 0.06414664536714554],
              [0.5483702719211578, -0.04979136399924755, 0.06517349928617477],
              [0.5472148656845093, -0.05017649382352829, 0.06530388444662094],
              [0.5441338419914246, -0.05017649382352829, 0.0663146823644638],
              [0.541437953710556, -0.05017649382352829, 0.06745342537760735],
              [0.5391174554824829, -0.05017649382352829, 0.06782390177249908],
              [0.5372015237808228, -0.05017649382352829, 0.06765850633382797],
              [0.5354478657245636, -0.04979136399924755, 0.06704223155975342],
              [0.530269205570221, -0.04979136399924755, 0.0633685290813446]])
}

# 传入 ground-truth mesh 和重建 mesh，计算F-score (0~1) 衡量重建效果的分数，越高说明越接近真实模型。
# mesh_gt: ground-truth（真实的）肿瘤网格模型。
# mesh_reconstructed: 重建的肿瘤网格模型（算法输出）。
def compute_f_score(mesh_gt, mesh_reconstructed):
    tau = TAU
    # 用于定义匹配点之间的距离容忍度（小于 tau 算匹配成功）。


    # 把网格中心平移到原点 (0,0,0)，方便比较两个网格。
    def center_mesh(mesh):
        m = copy.deepcopy(mesh)
        center = m.get_center()
        m.translate(-center)
        return m

    def preprocess(mesh):
    #  中心化 网格（上一步的函数）。
    #    均匀采样 3000 个点，得到稠密度适中的点云。
    # 返回 点云（用于后续点对点比较）。
        mesh = center_mesh(mesh)
        pcd = mesh.sample_points_uniformly(3000)
        return pcd
# 分别将 ground-truth 和 重建模型转为点云，并中心化。
    pcd_gt = preprocess(mesh_gt)
    pcd_reconstructed = preprocess(mesh_reconstructed)


    o3d.visualization.draw_geometries([pcd_gt, pcd_reconstructed])
    # 可视化 两个点云模型，帮助直观检查重建效果。调试用


    # Convert to numpy arrays for distance computation
    points_gt = np.asarray(pcd_gt.points)
    points_reconstructed = np.asarray(pcd_reconstructed.points)

    # Compute distances for precision
    dists_precision = o3d.geometry.KDTreeFlann(pcd_gt)
    # 用 ground-truth 点云创建 KD 树，加速最近邻搜索。后续求 precision 时使用。
    precision_count = 0
    for point in points_reconstructed:
        _, _, dist = dists_precision.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < tau:
            precision_count += 1
    precision = precision_count / len(points_reconstructed)
# 初始化计数器 precision_count。
# 遍历重建点云中的每个点。
# 对每个点，查找 GT 点云中最近的一个点，返回距离。
# 如果距离小于 tau，认为匹配成功，计数器加一。
# 最终 precision：precision = 匹配成功的重建点 / 总重建点数



# Precision	针对重建模型，重建点有多少比例匹配GT	      "我重建的东西有多少是真的？"
# Recall	针对 Ground-truth，GT中有多少被重建覆盖 	"我重建的东西覆盖了多少真的？"
# Precision 高，说明 "假阳性少"（重建的点基本都是对的）。
# Recall 高，说明 "假阴性少"（基本所有 GT 目标都被你找到）。



    # Compute distances for recall
    dists_recall = o3d.geometry.KDTreeFlann(pcd_reconstructed)
    recall_count = 0
    for point in points_gt:
        _, _, dist = dists_recall.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < tau:
            recall_count += 1
    recall = recall_count / len(points_gt)

    # Compute F-score
    f_score = (2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

    return f_score
# F-score 是 precision 和 recall 的调和平均值
# mesh_gt  +  mesh_reconstructed
#     │
# [中心化 & 均匀采样 3000 点]
#     │
#  得到点云 pcd_gt 和 pcd_reconstructed
#     │
#  [计算 Precision] (重建点到 GT 点)
#     │
#  [计算 Recall] (GT 点到重建点)
#     │
#  [F-score] = 2 * P * R / (P + R)
#     │
#   输出 F-score 分数



@click.command()
@click.option('--dataset_path',
              type=str,
              default='dataset_01-02-2024_22-15-10',
              help='Folder name in data directory containing experiment data')
@click.option('--use_glob', type=bool, default=False, help='Find datasets by glob')
@click.option('--combine', type=bool, default=False, help='Combine later datasets into one')
@click.option('--tumor', type=str, default='hemisphere', help='Tumor type to evaluate')
@click.option('--use_rerun', type=bool, default=False, help='Visualize with rerun')
@click.option('--use_cached_bbox_gt',
              type=bool,
              default=False,
              help='Use the cached bounding box for groundtruth')
@click.option('--use_cached_bbox', type=bool, default=False, help='Use the cached bounding box')


# python eval_dataset.py \
#   --dataset_path dataset_03-01-2024 \
#   --tumor hemisphere \
#   --use_glob False \
#   --combine True \
#   --use_rerun True \
#   --use_cached_bbox True \
#   --use_cached_bbox_gt True
    # ➡️ 这条命令的效果：
    # 使用 2024 年 3 月 1 日的数据集。
    # 针对 hemisphere 肿瘤类型评估。
    # 不批量查找。
    # 合并多个结果。
    # 开启 Rerun 3D 可视化。
    # 使用之前缓存的裁剪框，避免重复选择。


def main(dataset_path, use_glob, combine, tumor, use_rerun, use_cached_bbox_gt, use_cached_bbox):
    global EVAL_CROP, EVAL_CROP_GT
    if not use_cached_bbox_gt:
        EVAL_CROP_GT = {key: None for key in EVAL_CROP_GT}


    combined_tumor_recon = defaultdict(o3d.geometry.PointCloud)
    # 用来合并多个重建结果的容器（点云集合）。
    gt_scan = o3d.io.read_point_cloud(str(GT_PATH))
    # 加载 Ground-truth 扫描的点云。
    gt_tumors_pcd = {}
    gt_tumors_mesh = {}
# 存储处理后（如裁剪、上色）的 GT 点云和 mesh。

    for key in EVAL_CROP.keys():
        print(key)
        gt_scan_mesh = scan2mesh(gt_scan)
        gt_scan_mesh = mesh2polyroi(gt_scan_mesh, EVAL_CROP_GT[key], return_mesh=True)
        gt_scan_pcd = mesh2pcd(gt_scan_mesh)
        gt_tumors_mesh[key] = color_entity(gt_scan_mesh, color_map="rainbow")
        gt_tumors_pcd[key] = color_entity(gt_scan_pcd)
        #visualize_pcds([gt_tumors_pcd[key]])
# 针对每种肿瘤（如 hemisphere, crescent），从扫描数据生成 mesh。
# 按 GT 区域裁剪后，转为点云，进行上色。
# 存储在字典里，供后面评估使用。



    dataset_map = {}
    final_fscores_map = defaultdict(list)
    datasets = []
    if use_glob:
        for tumor in TUMOR_ID.keys():
            for algo in ALGOS:
                sets = glob.glob(str(SEEBELOW_DATA_PATH / "*{}*{}*".format(tumor, algo)))
                dataset_map[(tumor, algo)] = np.arange(len(datasets), len(datasets) + len(sets))
                datasets += [SEEBELOW_DATA_PATH / s for s in sets]

        f_scores = np.zeros((len(datasets), 2))
        print("Found {} datasets".format(len(datasets)))
    else:
        datasets = [SEEBELOW_DATA_PATH / dataset_path]
    if use_rerun:
        print("Init rerun")
        rr.init("seebelow_eval", spawn=False)
        rr.set_time_seconds("capture_time", 0)
    for i, dataset_path in enumerate(datasets):
        print(f"Evaluating {dataset_path.name}")
        palpations_cnt = 0

        recon_realtime = []
        tumor_pcd_without_CF = o3d.geometry.PointCloud()
        save_path = SEEBELOW_MESH_PATH.parent / "eval" / f"eval_{dataset_path.name}"
        save_path.mkdir(parents=True, exist_ok=True)

        if use_rerun:
            rerun_save_path = str(save_path / f"rerun_{dataset_path.name}.rrd")
            print("rerun saving to: ", rerun_save_path)
            rr.save(rerun_save_path)
        #search_history = np.load(dataset_path / "search_history.npy")
        #ani = HeatmapAnimation(search_history)
        #ani.visualize()
        timeseries = np.load(dataset_path / "timeseries.npy")
        palpations_cnt += timeseries[:]["palp_id"].max()
        roi_pcd = o3d.io.read_point_cloud(str(dataset_path / "roi.ply"))
        grid_pcd = o3d.io.read_point_cloud(str(dataset_path / "grid.ply"))
        recon_pth = dataset_path / "reconstruction.ply"
        tumor_pcd_with_CF = o3d.io.read_point_cloud(str(recon_pth))
        tumor_pcd_with_CF = color_entity(tumor_pcd_with_CF)
        tumor_pcd_with_CF_np = np.loadtxt(dataset_path / 'reconstruction.txt')
        tumor_pcd_with_CF_np_colors = color_entity(tumor_pcd_with_CF_np)

        # ground truth init
        cfg = yaml.safe_load(open(str(dataset_path / "config.yml")))
        tumor_type = cfg["tumor_type"]
        algo = cfg["algo"]
        if combine:
            combined_tumor_recon[TUMOR_ID[tumor_type]] += tumor_pcd_with_CF
            continue
        print(f"Evaluating tumor {tumor_type}")
        ground_truth_mesh_scan = gt_tumors_mesh[tumor_type]
        ground_truth_pcd_scan = gt_tumors_pcd[tumor_type]

        # log data
        if use_rerun:
            rr.log(
                "pcds/ground_truth_tumor",
                pcd_to_rr(
                    "gt_tumor",
                    np.asarray(ground_truth_pcd_scan.points),
                    colors=np.asarray(ground_truth_pcd_scan.colors),
                ),
            )
            rr.log(
                f"pcds/grid/{dataset_path.name}",
                pcd_to_rr("grid", np.asarray(grid_pcd.points)),
            )
            rr.log(
                f"pcds/roi/{dataset_path.name}",
                pcd_to_rr("roi", np.asarray(roi_pcd.points)),
            )
            recon_np = np.asarray(tumor_pcd_with_CF.points)
            rr.log(
                f"pcds/reconstructed_tumor/{dataset_path.name}",
                pcd_to_rr(
                    "recon_tumor",
                    recon_np,
                    colors=np.asarray(tumor_pcd_with_CF.colors),
                ),
            )
        collect_points = timeseries[:]["collect_points_flag"]
        collect_points_t, _ = np.where(collect_points == 1)
        O_p_E = timeseries[collect_points_t]["O_p_EE"]
        O_q_E = timeseries[collect_points_t]["O_q_EE"]
        init_O_p_surf = False
        init_O_p_f = False

        Fxyz = timeseries[:]["Fxyz"]
        O_p_E = timeseries[:]["O_p_EE"]
        using_force_control_flag = timeseries[:]["using_force_control_flag"]
        palp_state = timeseries[:]["palp_state"]

        return_t = np.array(np.where(palp_state)[0])
        top_t = np.where(np.diff(timeseries[:]["palp_state"], axis=0) == -2)[0]
        O_p_top = np.einsum("ijk->ik", timeseries[top_t]["O_p_EE"])
        O_q_top = np.einsum("ijk->ik", timeseries[top_t]["O_q_EE"])
        vecs_top = np.zeros_like(O_p_top)
        for i in range(len(O_q_top)):
            vecs_top[i, :] = quat2mat(O_q_top[i].flatten())[:3, 2]  # zaxis is vector direction

        O_p_surf_t = np.where(np.diff(timeseries[:]["palp_id"], axis=0) == 1)[0]
        O_p_surf = np.einsum("ijk->ik", timeseries[O_p_surf_t]["O_p_EE"])
        O_p_f_t = np.where(np.diff(timeseries[:]["using_force_control_flag"], axis=0) == 1)[0]
        O_p_f = np.einsum("ijk->ik", timeseries[O_p_f_t]["O_p_EE"])
        if O_p_f.shape == O_p_surf.shape:
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(O_p_f)
            tumor_pcd_without_CF += final_pcd
            if use_rerun:
                rr.log(
                    f"pcds/positioning_end/{dataset_path.name}",
                    pcd_to_rr("positioning_end", O_p_f),
                )
                '''
                rr.log(
                    f"pcds/top_poses/{dataset_path.name}",
                    vectors_to_rr(
                        "top_poses",
                        origins=O_p_top,
                        vectors=vecs_top,
                    ),
                )
                '''
        if use_rerun:
            collect_points_i = 0
            for t in tqdm(range(len(timeseries))):
                rr.set_time_seconds("capture_time", t / 50)
                Fxyz = timeseries[t]["Fxyz"]
                O_p_E = timeseries[t]["O_p_EE"]
                O_q_E = timeseries[t]["O_q_EE"]
                collect_points_flag = timeseries[t]["collect_points_flag"]
                using_force_control_flag = timeseries[t]["using_force_control_flag"]
                palp_state = timeseries[t]["palp_state"]
                palp_id = timeseries[t]["palp_id"]
                collect_points_flag = timeseries[t]["collect_points_flag"]
                stiffness_fz = 0
                stiffness_exp = 0
                if use_rerun and collect_points_flag:
                    #recon_realtime.append(O_p_E.copy().flatten())
                    #recon_realtime_np = np.array(recon_realtime)
                    rr.log(
                        f"pcds/recon_realtime/{dataset_path.name}",
                        pcd_to_rr(
                            "recon_realtime",
                            tumor_pcd_with_CF_np[:collect_points_i],
                            colors=tumor_pcd_with_CF_np_colors[:collect_points_i],
                        ),
                    )
                    collect_points_i += 1
                if palp_state == PalpateState.PALPATE:
                    if not init_O_p_surf:
                        O_p_surf = O_p_E
                        init_O_p_surf = True
                    if using_force_control_flag and not init_O_p_f:
                        O_p_f = O_p_E
                        stiffness_fz = Fxyz[0, 2] / np.linalg.norm(O_p_f - O_p_surf)
                        stiffness_fz /= PALP_CONST.stiffness_normalization
                        # stiffness_exp = timeseries[t]["stiffness"]
                        init_O_p_f = True
                    T = np.eye(4)
                    T[:3, :3] = quat2mat(O_q_E.flatten())
                    T[:3, 3] = O_p_E.flatten()
                    rr_tf = TranslationAndMat3x3(translation=T[:3, 3], mat3x3=T[:3, :3])
                    rr.log("pcds/eef_pose", rr.Transform3D(transform=rr_tf))
                    rr.log("force/x", rr.Scalar(Fxyz[0, 0]))
                    rr.log("force/y", rr.Scalar(Fxyz[0, 1]))
                    rr.log("force/z", rr.Scalar(Fxyz[0, 2]))
                    if collect_points_flag:
                        # rr.log("stiffness/experiment", rr.Scalar(stiffness_exp))
                        rr.log("stiffness/Fz", rr.Scalar(stiffness_fz))
                else:
                    init_O_p_surf = False
                    init_O_p_f = False
                '''
                palp_id = min(len(search_history) - 1, palp_id)
                grid = search_history[palp_id]["grid"]
                rr.log("search_grid", rr.Tensor(grid, dim_names=("batch", "X", "Y")))
                '''
        tumor_mesh_without_CF = color_entity(scan2mesh(tumor_pcd_without_CF))
        tumor_mesh_with_CF = color_entity(scan2mesh(tumor_pcd_with_CF))
        if not use_cached_bbox:
            EVAL_CROP = {key: None for key in EVAL_CROP}
        tumor_mesh_without_CF = mesh2polyroi(
            tumor_mesh_without_CF,
            polybox_pts=EVAL_CROP[tumor_type],
            return_mesh=True,
        )
        tumor_mesh_with_CF = mesh2polyroi(
            tumor_mesh_with_CF,
            polybox_pts=EVAL_CROP[tumor_type],
            return_mesh=True,
        )
        if use_rerun:
            rr.log(
                "mesh/without_CF",
                mesh_to_rr("mesh_without_CF",
                           verticies=np.asarray(tumor_mesh_without_CF.vertices),
                           vertex_normals=np.asarray(tumor_mesh_without_CF.vertex_normals),
                           vertex_colors=np.asarray(tumor_mesh_without_CF.vertex_colors),
                           triangles=np.asarray(tumor_mesh_without_CF.triangles)))
            rr.log(
                "mesh/with_CF",
                mesh_to_rr("mesh_with_CF",
                           verticies=np.asarray(tumor_mesh_with_CF.vertices),
                           vertex_normals=np.asarray(tumor_mesh_with_CF.vertex_normals),
                           vertex_colors=np.asarray(tumor_mesh_with_CF.vertex_colors),
                           triangles=np.asarray(tumor_mesh_with_CF.triangles)))
            rr.log(
                "mesh/ground_truth_mesh",
                mesh_to_rr("ground_truth_mesh",
                           verticies=np.asarray(ground_truth_mesh_scan.vertices),
                           vertex_normals=np.asarray(ground_truth_mesh_scan.vertex_normals),
                           vertex_colors=np.asarray(ground_truth_mesh_scan.vertex_colors),
                           triangles=np.asarray(ground_truth_mesh_scan.triangles)))
        print(f"tumor_mesh_without_CF: {len(tumor_mesh_without_CF.vertices)}")
        print(f"tumor_mesh_with_CF: {len(tumor_mesh_with_CF.vertices)}")
        # compute f-scores
        f_score_without_CF = compute_f_score(ground_truth_mesh_scan, tumor_mesh_without_CF)
        f_score_with_CF = compute_f_score(ground_truth_mesh_scan, tumor_mesh_with_CF)
        # print(f"F-score sanity check: {f_score_sanity_check}")
        print(f"F-score with CF: {f_score_with_CF}")
        print(f"F-score without CF: {f_score_without_CF}")
        yaml.dump(
            {
                "f_score_with_CF": f_score_with_CF,
                "f_score_without_CF": f_score_without_CF,
                "tau": TAU,
            },
            open(save_path / "f_score.yaml", "w"),
        )

        o3d.io.write_triangle_mesh(str(save_path / "mesh_without_CF.ply"), tumor_mesh_without_CF)
#         将 不带颜色滤波 (without_CF) 的肿瘤重建网格保存为 .ply 文件。
#         save_path 是本次评估结果存储文件夹。
#         .ply 格式：标准 3D 点云/网格文件格式，方便后续查看、分析。

        o3d.io.write_triangle_mesh(str(save_path / "mesh_with_CF.ply"), tumor_mesh_with_CF)
        
        o3d.io.write_triangle_mesh(str(save_path / "mesh_gt_scan.ply"), ground_truth_mesh_scan)

        if use_glob:
            f_scores[i, 0] = f_score_with_CF
            f_scores[i, 1] = f_score_without_CF
            # if (input("Do you want to eval this run? (y/n) ") == "y"):
            print(f"tumor {tumor_type}, algo {algo}")
            final_fscores_map[(tumor_type, algo)].append(i)
            print(
                "current # of samples for this category: ",
                len(final_fscores_map[(tumor_type, algo)]),
            )
    if combine:
        for tumor, i in TUMOR_ID.items():
            combined_mesh = color_entity(scan2mesh(combined_tumor_recon[i]))
            combined_mesh_roi = mesh2polyroi(combined_mesh,
                                             polybox_pts=EVAL_CROP[tumor],
                                             return_mesh=True)
            ground_truth_mesh_scan = gt_tumors_scan_mesh[i]
            o3d.visualization.draw_geometries([combined_mesh_roi])
            fscore = compute_f_score(ground_truth_mesh_scan, combined_mesh_roi)
            print(f"Combined F-score for all experiments {tumor}: {fscore}")
    elif use_glob:
        for tumor in TUMOR_ID.keys():
            for algo in ALGOS:
                fscores_ids = final_fscores_map[(tumor, algo)]
                print("MEAN {},{} #{}: {}".format(
                    tumor,
                    algo,
                    len(fscores_ids),
                    np.mean(f_scores[fscores_ids], axis=0),
                ))
                print("MAX {},{} #{}: {}".format(
                    tumor,
                    algo,
                    len(fscores_ids),
                    np.max(f_scores[fscores_ids], axis=0),
                ))
                print("MIN {},{} #{}: {}".format(
                    tumor,
                    algo,
                    len(fscores_ids),
                    np.min(f_scores[fscores_ids], axis=0),
                ))

    if use_rerun:
        rr.disconnect()
        print("Saved rerun!")


if __name__ == "__main__":
    main()

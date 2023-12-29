from pathlib import Path
import numpy as np
import rpal


def array2constant(var_name: str, arr: np.ndarray) -> str:
    return f"{var_name.upper()} = np.array({arr.tolist()})"


# spacemouse vendor and product id
SPACEM_VENDOR_ID = 9583
SPACEM_PRODUCT_ID = 50741

# useful paths
RPAL_PKG_PATH = Path(rpal.__file__).parent.absolute()
RPAL_CFG_PATH = Path(rpal.__file__).parent.absolute() / "config"
RPAL_MESH_PATH = Path(rpal.__file__).parent.absolute() / "meshes"
RPAL_CKPT_PATH = Path(rpal.__file__).parent.absolute() / ".ckpts"

# controllers
RPAL_HYBRID_POSITION_FORCE = "RPAL_HYBRID_POSITION_FORCE"
OSC_CTRL_TYPE = "OSC_POSE"
OSC_DELTA_CFG = "osc-pose-controller-delta.yml"
OSC_ABSOLUTE_CFG = "osc-pose-controller.yml"
JNT_POSITION_CFG = "joint-position-controller.yml"
BASE_CALIB_FOLDER = RPAL_CFG_PATH / "base_camera_calib"
PAN_PAN_FORCE_CFG = "pan-pan-force.yml"

# surface scan
SURFACE_SCAN_PATH = RPAL_PKG_PATH / "meshes" / "tumors_gt_12-27-2023_20-16-38.ply"

# tumor color thresholding for ground truth collection
TUMOR_HSV_THRESHOLD = (np.array([80, 136, 3]), np.array([115, 255, 19]))

BBOX_PHANTOM = np.array(
    [
        [0.501941544576342, -0.05161805081947522, 0.5774690032616651],
        [0.5034256408649196, -0.05167661429723301, -0.443162425136426],
        [0.49413665553622205, 0.04050607695706589, 0.5774523681515825],
        [0.5808408971658352, -0.04493356316916096, 0.5775833469573991],
        [0.5745201044142929, 0.04713200112962236, -0.4430647165507745],
        [0.5730360081257153, 0.04719056460738015, 0.5775667118473166],
        [0.5823249934544128, -0.04499212664691875, -0.44304808144069185],
        [0.49562075182479964, 0.0404475134793081, -0.44317906024650866],
    ]
)

GT_SCAN_POSE = np.array(
    [
        0.5174325704574585,
        -0.0029695522971451283,
        0.25471308946609497,
        -0.929806649684906,
        0.36692020297050476,
        -0.025555845350027084,
        0.01326837856322527,
    ]
)


BBOX_ROI = np.array(
    [
        [0.506196825331111, -0.043235306355001196, 0.5804695750208919],
        [0.5066164297007841, -0.04348050754514507, -0.4311007035987048],
        [0.4928122028439124, 0.020206739021827125, 0.5804486448773705],
        [0.566811847214656, -0.030447105067965306, 0.580491618616227],
        [0.5538468290971307, 0.03274973911871914, -0.43109959014689136],
        [0.5534272247274575, 0.03299494030886301, 0.5804706884727056],
        [0.5672314515843292, -0.03069230625810918, -0.4310786600033698],
        [0.49323180721358556, 0.01996153783168325, -0.43112163374222634],
    ]
)


# interpolation
STEP_FAST = 100
STEP_SLOW = 2000

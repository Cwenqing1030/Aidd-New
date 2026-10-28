"""数据生成和训练共用的默认参数；直接在这里修改。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

# 三元超图与 SIR 数据；拓扑类型可选 ER、SF、WS。
NETWORK_TYPE = "WS"
N = 50
E = 200
SF_ALPHA = 1.0          # 排名权重的指数；不是实际超度分布的拟合指数
WS_REWIRE_PROB = 0.1    # 重连概率；WS 要求 E 是 N 的整数倍
NUM_TRAJECTORIES = 1000
STEPS_PER_TRAJECTORY = 5
INITIAL_INFECTED = N // 5
BETA = 0.5
GAMMA = 0.5
SEED = 42

# 网络文件共用一个目录，动力学名称进入各自的数据文件名。
NETWORK_PARAMETERS = {
    "ER": {},
    "SF": {"alpha": SF_ALPHA},
    "WS": {"rewire_prob": WS_REWIRE_PROB},
}[NETWORK_TYPE]
_network_suffix = {
    "ER": "",
    "SF": f"_alpha{SF_ALPHA:g}",
    "WS": f"_p{WS_REWIRE_PROB:g}",
}[NETWORK_TYPE]
NETWORK_NAME = f"{NETWORK_TYPE}3_N{N}_E{E}{_network_suffix}_seed{SEED}"
DYNAMICS_NAME = "SIR"
DATA_DIR = ROOT / "data" / NETWORK_NAME
OUTPUT_ROOT = ROOT / "outputs"

# 训练默认值，也可以通过 train.py 的命令行参数覆盖
DEVICE = "auto"  # auto、cpu、cuda 或 cuda:0/cuda:1 等逻辑设备编号
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EPOCHS = 10
BATCH_SIZE = 64
FEATURE_DIM = 128
PAIR_CHUNK_SIZE = 4096
LR_DYN = 0.001
LR_STRUCTURE = 0.004
LAMBDA_SPARSE = 0.0001
TEMPERATURE = 1.0
INIT_EDGE_PROB = 0.001
INIT_STD = 0.1
GUMBEL_HARD = False

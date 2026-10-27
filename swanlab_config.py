"""SwanLab 记录配置；服务器安装 swanlab>=0.10.0，并用 swanlab login 登录。"""

PROJECT = "HyCoRD"
WORKSPACE = "HKFGNo1"     # 组织的 username
EXPERIMENT_NAME = None    # None 按数据名、真实超边数量和训练种子命名
MODE = "online"           # online 上传；offline 仅保存日志；disabled 关闭记录
LOG_DIR_NAME = "swanlog"   # 日志放在训练 output_dir 的这个子目录

# API Key 通过服务器的 swanlab login 或 SWANLAB_API_KEY 环境变量提供。

set -x

# Step4 Update daily config file
config_file=midpage_e2e_train.json
# Step5 Start training task
# 基础配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 关键修改 ▼▼▼
export NCCL_SOCKET_IFNAME=lo       # 强制绑定到本地 IPv4 环回接口
export GLOO_SOCKET_IFNAME=lo       # Gloo 后端同样绑定
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand（避免走 RDMA）
export NCCL_P2P_DISABLE=1          # 禁用点对点直连
export DS_TRANSPORT_TCP=1          # DeepSpeed 强制使用 TCP
export FI_SHM_DISABLE=1            # 禁用共享内存传输
export PYTORCH_DISTRIBUTED_BACKEND=gloo  # 使用 Gloo 而非 NCCL（更稳定）
# 关键修改 ▲▲▲

# 调试输出
export NCCL_DEBUG=INFO
export GLOO_DEBUG=1


python  train_gr.py --config gr_train.json
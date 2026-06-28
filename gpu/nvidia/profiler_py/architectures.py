import os

arch_map = {
    "NVIDIA A100-SXM4-40GB": {"bw": 1555.0, "cores_sm": 64},
    "NVIDIA A100-SXM4-80GB": {"bw": 1935.0, "cores_sm": 64},
    "NVIDIA H100 80GB HBM3":  {"bw": 3350.0, "cores_sm": 128}
}
    
# smol-moe

**Heavily** under construction, just making it public to share notes with a few colleagues 

Run code via:
```
NCCL_ALGO=Tree NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0  NCCL_IB_DISABLE=1 NCCL_ALGO=Ring  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' CUDA_VISIBLE_DEVICES="1,2,3,4" accelerate launch --mixed_precision bf16 tflop_tester.py --precision bf16
```

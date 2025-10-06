# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal example of training with FP8 precision using FSDP2 via Accelerate.
This example demonstrates how to use torchao's Float8LinearConfig with Accelerate's AORecipeKwargs.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, FullyShardedDataParallelPlugin, TorchDynamoPlugin, set_seed, DistributedDataParallelKwargs, DataLoaderConfiguration
from utils import PerformanceTracker, create_collate_fn, get_dataset, get_model_flops_per_token
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from datasets import load_from_disk

# torch.backends.cuda.enable_triton_flash_attention(True)
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(False)

torch.cuda.cudagraphs_enabled = True

# torch.backends.cuda.matmul.allow_tf32 = True 
# torch.set_float32_matmul_precision('medium')

WARMUP_STEPS = 10
TRACE_DIR = "tf32_matmul_precision"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-length", type=int, default=4096, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to train for")
    parser.add_argument("--precision", type=str, default="fp8", choices=["fp8", "bf16"], help="Precision to train in")
    parser.add_argument("--log-with", type=str, default="wandb", help="Log with wandb or tensorboard")

    return parser.parse_args()

def get_dataset_offline(path="tokenized_tinystories"):
    dataset = load_from_disk(path)
    dataset.set_format(type="torch", columns=["input_ids", "labels", "shift_labels", "position_ids"])
    return dataset


def main():
    """
    Main function to train the model.
    """
    set_seed(42)

    args = parse_args()

    # fsdp2_plugin = FullyShardedDataParallelPlugin(
    #     fsdp_version=2,
    #     cpu_ram_efficient_loading=False,  # CPU RAM efficient loading CANNOT work with fp8 torchao
    #     auto_wrap_policy="transformer_based_wrap",
    #     transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
    #     reshard_after_forward=False,
    # )
    # fsdp2_plugin.set_mixed_precision(args.precision)

    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        use_regional_compilation=False,  # We use regional compilation to compile the model way faster
        fullgraph=False,
    )

    ddp_plugin = DistributedDataParallelKwargs(find_unused_parameters=True)


    kwargs = []
    kwargs=[ddp_plugin]

    accelerator = Accelerator(
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs,
        log_with=args.log_with,
        gradient_accumulation_steps=1,
        
    )

    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained("./config.json", use_cache=False),
        torch_dtype=torch.bfloat16,
        attn_implementation="flex_attention"
    )

    model.model.embed_tokens = torch.compile(model.model.embed_tokens)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    # dataset = get_dataset(tokenizer, args.sequence_length, accelerator)
    dataset = get_dataset_offline()
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=create_collate_fn(), pin_memory=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    accelerator.wait_for_everyone()

    model.train()

    total_num_steps = 200
    model_flops_per_token = get_model_flops_per_token(model, args.sequence_length)
    performance_tracker = PerformanceTracker(warmup_steps=5)

    # Setup profiler if enabled
    # profiler_context = None
    # if accelerator.is_main_process:
    #     # Create profiler schedule: skip warmup, then profile for specified steps
    #     profiler_schedule = schedule(
    #         skip_first=WARMUP_STEPS,  # Skip warmup steps
    #         wait=1,  # Wait 1 step after warmup
    #         warmup=100,  # Warmup profiler for 2 steps
    #         active=10,  # Profile for this many steps
    #         repeat=1  # Only profile once
    #     )
        
    #     profiler_context = profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=profiler_schedule,
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(TRACE_DIR),
    #         record_shapes=True,  # Record tensor shapes
    #         profile_memory=True,  # Track memory usage
    #         with_stack=True,  # Record source code information
    #         with_flops=True,  # Estimate FLOPs
    #     )
    #     accelerator.print(f"Profiler enabled")
    #     accelerator.print(f"Results will be saved to: {TRACE_DIR}")
    #     accelerator.print(f"View with: tensorboard --logdir={TRACE_DIR}")

    # if profiler_context:
    #     profiler_context.__enter__()
    import time
    prefetch_stream = torch.cuda.Stream()
    next_batch = None
    # if accelerator.is_main_process:
    start_time = time.time()
    len_inputs = 0
    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break
        with torch.cuda.stream(prefetch_stream):
            next_batch = {k: v.to(device=accelerator.device, non_blocking=True)
                      for k, v in batch.items()}
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        batch = next_batch
        # with record_function("forward"):
        if (step + 1) % 8 != 0:
            with model.no_sync():
                outputs = model(**batch)
                loss = outputs.loss / 8
        else:
            outputs = model(**batch)
            loss = outputs.loss / 8
        # with record_function("backward"):
        accelerator.backward(loss)
        # with record_function("optimizer_step"):
        if (step + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
        # if profiler_context:
        #     profiler_context.step()

    # if accelerator.is_main_process:
    end_time = time.time()
    print(f'Time taken for training: {end_time-start_time}')


    accelerator.wait_for_everyone()
    accelerator.end_training()
    # if profiler_context:
    #     profiler_context.__exit__(None, None, None)

    # accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
"""
Benchmark script for model speed and GPU usage metrics.

Measures throughput (images/sec), latency (ms/image), and GPU memory usage
for any backbone, with optional checkpoint loading.

Usage:
    # Benchmark a backbone without weights (random init):
    python benchmark.py --network r50

    # Benchmark with a trained checkpoint:
    python benchmark.py --network r50 -w path/to/checkpoint.pt

    # Customize batch sizes and image size:
    python benchmark.py --network ir50 --batch-sizes 1,8,32,64,128 --img-size 112

    # Use FP16:
    python benchmark.py --network r50 --fp16

    # Compare all available backbones:
    python benchmark.py --all
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from backbones import get_model

ALL_NETWORKS = [
    "ir50", "ir100",
    "r50", "r101",
    "swinb", "vitb",
    "mobilenetv3s", "mobilenetv3l",
    "efficientnetb0", "efficientnetv2s", "efficientnetv2m", "efficientnetv2l",
]

DEFAULT_BATCH_SIZES = [1, 8, 32, 64]


def get_gpu_memory_mb():
    """Return (allocated, reserved) GPU memory in MB."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return allocated, reserved


def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model(model, input_size, batch_sizes, warmup_iters=10,
                    bench_iters=50, use_fp16=False, device="cuda"):
    """Benchmark a model across multiple batch sizes.

    Returns a list of dicts with metrics per batch size.
    """
    model.eval()
    model.to(device)
    results = []

    dtype = torch.float16 if use_fp16 else torch.float32

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        dummy = torch.randn(bs, 3, input_size, input_size,
                            device=device, dtype=dtype)

        # Warmup
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
            for _ in range(warmup_iters):
                _ = model(dummy)
        torch.cuda.synchronize()

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
            start_event.record()
            for _ in range(bench_iters):
                _ = model(dummy)
            end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        total_images = bs * bench_iters
        ms_per_batch = elapsed_ms / bench_iters
        ms_per_image = elapsed_ms / total_images
        images_per_sec = total_images / (elapsed_ms / 1000.0)

        alloc_mb, reserved_mb = get_gpu_memory_mb()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results.append({
            "batch_size": bs,
            "total_ms": elapsed_ms,
            "ms_per_batch": ms_per_batch,
            "ms_per_image": ms_per_image,
            "images_per_sec": images_per_sec,
            "gpu_allocated_mb": alloc_mb,
            "gpu_reserved_mb": reserved_mb,
            "gpu_peak_mb": peak_mb,
        })

    return results


def print_results(network_name, results, total_params, trainable_params,
                  input_size, use_fp16):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 75}")
    print(f"  Model: {network_name}")
    print(f"  Input: {input_size}x{input_size}  |  "
          f"Params: {total_params / 1e6:.2f}M  |  "
          f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"{'=' * 75}")
    print(f"  {'Batch':>6}  {'ms/batch':>10}  {'ms/image':>10}  "
          f"{'img/sec':>10}  {'GPU alloc':>10}  {'GPU peak':>10}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 10}  {'-' * 10}  {'-' * 10}")

    for r in results:
        print(f"  {r['batch_size']:>6}  "
              f"{r['ms_per_batch']:>9.2f}ms  "
              f"{r['ms_per_image']:>9.3f}ms  "
              f"{r['images_per_sec']:>9.1f}  "
              f"{r['gpu_allocated_mb']:>8.1f}MB  "
              f"{r['gpu_peak_mb']:>8.1f}MB")

    print()


def run_benchmark(network, weight_path, img_size, batch_sizes, use_fp16,
                  warmup, iters, device):
    """Load a model, optionally load weights, and benchmark it."""
    print(f"\nLoading model: {network} ...", end=" ", flush=True)
    try:
        model = get_model(network, dropout=0.0, fp16=False, num_features=512)
    except TypeError:
        # Some backbones don't accept these kwargs
        model = get_model(network)

    if weight_path:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict_backbone"])
        print(f"weights loaded from {weight_path}")
    else:
        print("(random init)")

    total_params, trainable_params = count_parameters(model)

    results = benchmark_model(
        model, img_size, batch_sizes,
        warmup_iters=warmup, bench_iters=iters,
        use_fp16=use_fp16, device=device,
    )

    print_results(network, results, total_params, trainable_params,
                  img_size, use_fp16)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model inference speed and GPU usage")
    parser.add_argument("--network", type=str, default=None,
                        help="Backbone name (e.g. r50, ir50, swinb, vitb)")
    parser.add_argument("-w", dest="weight", type=str, default=None,
                        help="Path to checkpoint (.pt) file")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,8,32,64)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of timed iterations per batch size")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 (mixed precision) inference")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all available backbones")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = "cuda"
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes else DEFAULT_BATCH_SIZES
    )

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"\nGPU: {gpu_name}  ({gpu_total_mb:.0f} MB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    networks = ALL_NETWORKS if args.all else [args.network]

    if not args.all and args.network is None:
        parser.error("Provide --network or --all")

    user_set_img_size = args.batch_sizes is not None or args.img_size != 224

    for net in networks:
        img_size = args.img_size
        # iresnet uses 112x112 by default; auto-adjust when running --all
        if net and net.startswith("ir") and args.img_size == 224:
            if args.all:
                img_size = 112
                print(f"  Note: auto-setting img_size=112 for {net}")
            else:
                print(f"  Note: {net} typically uses 112x112 input; "
                      f"pass --img-size 112 to match training config")

        try:
            run_benchmark(net, args.weight, img_size, batch_sizes,
                          args.fp16, args.warmup, args.iters, device)
        except Exception as e:
            print(f"\n  FAILED to benchmark {net}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()

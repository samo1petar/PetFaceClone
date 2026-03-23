"""Convert ONNX model to TensorRT engine for Jetson Orin Nano deployment.

Run this script ON the Jetson device (TensorRT engines are device-specific).

Usage:
    # FP16 (recommended - good speed/accuracy trade-off):
    python export_tensorrt.py \
        --onnx model.onnx \
        --output model_fp16.engine \
        --fp16

    # INT8 with calibration (fastest, needs calibration images):
    python export_tensorrt.py \
        --onnx model.onnx \
        --output model_int8.engine \
        --int8 \
        --calib-dir /path/to/calibration/images \
        --calib-count 500

    # Or use trtexec directly (no Python needed):
    trtexec --onnx=model.onnx --saveEngine=model_fp16.engine \
            --fp16 --workspace=1024
"""

import argparse
import os
import sys


def export_with_trtexec(args):
    """Export using trtexec command-line tool (most reliable method)."""
    cmd_parts = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        f"--workspace={args.workspace}",
    ]

    if args.fp16:
        cmd_parts.append("--fp16")
    if args.int8:
        cmd_parts.append("--int8")
        if args.calib_cache:
            cmd_parts.append(f"--calib={args.calib_cache}")

    # Set input shape
    cmd_parts.append(f"--optShapes=input:1x3x{args.img_size}x{args.img_size}")
    cmd_parts.append(f"--minShapes=input:1x3x{args.img_size}x{args.img_size}")
    cmd_parts.append(f"--maxShapes=input:{args.max_batch}x3x{args.img_size}x{args.img_size}")

    cmd = " ".join(cmd_parts)
    print(f"Running: {cmd}")
    os.system(cmd)


def export_with_python_api(args):
    """Export using TensorRT Python API with optional INT8 calibration."""
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt not found. Install with:")
        print("  pip install tensorrt")
        print("Or use trtexec instead: --method trtexec")
        sys.exit(1)

    import numpy as np

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Loading ONNX: {args.onnx}")
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            sys.exit(1)

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 20))

    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, args.img_size, args.img_size),
        opt=(1, 3, args.img_size, args.img_size),
        max=(args.max_batch, 3, args.img_size, args.img_size),
    )
    config.add_optimization_profile(profile)

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        else:
            print("WARNING: FP16 not supported on this platform, using FP32")

    if args.int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 enabled")

            if args.calib_dir:
                calibrator = ImageCalibrator(
                    args.calib_dir, args.calib_count, args.img_size, args.calib_cache,
                )
                config.int8_calibrator = calibrator
            elif args.calib_cache and os.path.exists(args.calib_cache):
                print(f"Using cached calibration: {args.calib_cache}")
            else:
                print("WARNING: INT8 without calibration data, accuracy may suffer")
        else:
            print("WARNING: INT8 not supported on this platform")

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Failed to build engine")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nExported to: {args.output}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  FP16: {args.fp16}")
    print(f"  INT8: {args.int8}")


class ImageCalibrator:
    """INT8 calibrator using real images for TensorRT."""

    def __init__(self, image_dir, num_images, img_size, cache_file):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt required for calibration")
        import glob

        import cv2
        import numpy as np

        self.cache_file = cache_file
        self.batch_size = 1
        self.img_size = img_size
        self.current_index = 0

        # Collect image paths
        exts = ["*.jpg", "*.jpeg", "*.png"]
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        self.image_paths = self.image_paths[:num_images]
        print(f"Calibration: {len(self.image_paths)} images from {image_dir}")

        # Allocate device buffer
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.device_input = cuda.mem_alloc(
            self.batch_size * 3 * img_size * img_size * 4  # float32
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        import cv2
        import numpy as np
        import pycuda.driver as cuda

        path = self.image_paths[self.current_index]
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.ascontiguousarray(img[np.newaxis, ...])  # add batch dim

        cuda.memcpy_htod(self.device_input, img)
        self.current_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)


def benchmark_engine(engine_path, img_size, num_runs=100):
    """Benchmark TensorRT engine inference speed."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        import time
    except ImportError:
        print("Cannot benchmark: tensorrt/pycuda not available")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("input", (1, 3, img_size, img_size))

    # Allocate buffers
    input_data = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    output_data = np.empty((1, 256), dtype=np.float32)  # embedding size

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    stream = cuda.Stream()

    # Warmup
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    cuda.memcpy_dtoh(output_data, d_output)
    norm = np.linalg.norm(output_data, axis=1)

    times = np.array(times)
    print(f"\nTensorRT Inference Benchmark ({num_runs} runs):")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {times.min():.2f} ms")
    print(f"  Max: {times.max():.2f} ms")
    print(f"  Output shape: {output_data.shape}")
    print(f"  L2 norm: {norm[0]:.4f} (should be ~1.0)")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output TensorRT engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-batch", type=int, default=4, help="Max batch size for dynamic shapes")
    parser.add_argument("--workspace", type=int, default=1024, help="Workspace size in MB")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory with calibration images for INT8")
    parser.add_argument("--calib-count", type=int, default=500,
                        help="Number of calibration images")
    parser.add_argument("--calib-cache", type=str, default="calibration.cache",
                        help="Calibration cache file")
    parser.add_argument("--method", choices=["python", "trtexec"], default="python",
                        help="Export method")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    args = parser.parse_args()

    if not args.fp16 and not args.int8:
        print("NOTE: No precision flag set, defaulting to FP16")
        args.fp16 = True

    if args.method == "trtexec":
        export_with_trtexec(args)
    else:
        export_with_python_api(args)

    if args.benchmark:
        benchmark_engine(args.output, args.img_size)


if __name__ == "__main__":
    main()

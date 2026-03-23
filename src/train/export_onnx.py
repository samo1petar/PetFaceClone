"""Export trained model to ONNX format for Jetson deployment.

Usage:
    uv run python export_onnx.py \
        --checkpoint ../../outputs/exp10_effnetv2b0/model_best.pt \
        --backbone tf_efficientnetv2_b0.in1k \
        --embedding-size 256 \
        --output ../../outputs/exp10_effnetv2b0/model.onnx
"""

import argparse

import torch
import torch.nn.functional as F

from model import EmbeddingBackbone


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_b0.in1k")
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    # Load model
    backbone = EmbeddingBackbone(args.backbone, args.embedding_size, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["state_dict_backbone"])
    backbone.eval()

    if args.fp16:
        backbone = backbone.half()
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float16)
    else:
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    print(f"Loaded: {args.checkpoint}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Embedding: {args.embedding_size}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, AUC: {ckpt.get('auc', '?')}")

    # Export to ONNX
    torch.onnx.export(
        backbone,
        dummy_input,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )

    # Verify
    import onnx
    model = onnx.load(args.output)
    onnx.checker.check_model(model)

    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nExported to: {args.output}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Opset: {args.opset}")
    print(f"  FP16: {args.fp16}")

    # Quick verification of ONNX output
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(args.output)
    test_input = np.random.randn(1, 3, args.img_size, args.img_size).astype(
        np.float16 if args.fp16 else np.float32
    )
    result = session.run(None, {"input": test_input})
    embedding = result[0]
    norm = np.linalg.norm(embedding, axis=1)
    print(f"  Output shape: {embedding.shape}")
    print(f"  L2 norm: {norm[0]:.4f} (should be ~1.0)")


if __name__ == "__main__":
    main()

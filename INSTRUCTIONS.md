# PetFace Installation and Usage Instructions

This guide explains how to install PetFace and run it on your custom data located in `dataset/after_4_bis/`.

## 1. Installation

### Option A: Docker (Recommended)

**Step 1: Pull the Docker image**

```bash
docker pull pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
```

**Step 2: Run the container**

```bash
docker run -it --gpus all --shm-size 64G \
    -v /home/alfred/Projects/PetFace:/workspace/ \
    pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime bash
```

> Note: Replace `/home/alfred/Projects/PetFace` with your actual repository path.

**Step 3: Inside the container, install dependencies**

```bash
# Install Python packages
pip install easydict mxnet onnx scikit-learn timm tensorboard scipy==1.7.3

# Install system dependencies
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
apt-get install ffmpeg libsm6 libxext6 git -y

# Install additional Python packages
pip install opencv-python-headless==4.5.5.64 pandas==1.3.5
pip install seaborn matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install scikit-image
```

Or simply run:
```bash
bash install.sh
```

**Step 4: Run inference inside the container**

```bash
cd /workspace

# Generate CSV for your data
echo "filename,label" > dataset/reidentification.csv
for dir in dataset/after_4_bis/*/; do
    id=$(basename "$dir")
    for img in "$dir"*.jpg; do
        [ -f "$img" ] && echo "${img#dataset/},$id" >> dataset/reidentification.csv
    done
done

# Create results directory
mkdir -p results

# Run re-identification
CUDA_VISIBLE_DEVICES=0 python3 src/reidentification.py \
  -m arcface \
  -w pretrained/dog.pt \
  -i dataset/reidentification.csv \
  -o results/reid_results.csv \
  -b dataset

# Compute accuracy
python3 src/compute_topk_acc.py --topk 5 -i results/reid_results.csv
```

### Option B: pip Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch 1.12.0 with CUDA 11.3
- timm, transformers, opencv-python
- scikit-learn, pandas, numpy

---

## 2. Prepare Your Data

Your data in `dataset/after_4_bis/` is structured as:
```
dataset/after_4_bis/
├── 0/
│   ├── 0.0.jpg
│   ├── 0.1.jpg
│   └── ...
├── 1/
│   ├── 1.0.jpg
│   └── ...
└── ... (folders 0 to 1425)
```

### Create CSV Files

PetFace requires CSV files to specify which images to process. Create these files:

#### For Re-identification (`reidentification.csv`)

Format: `filename,label`

```bash
# Generate reidentification.csv automatically
cd /home/alfred/Projects/PetFace

# Create the CSV file
echo "filename,label" > dataset/reidentification.csv
for dir in dataset/after_4_bis/*/; do
    id=$(basename "$dir")
    for img in "$dir"*.jpg; do
        if [ -f "$img" ]; then
            relpath="${img#dataset/}"
            echo "$relpath,$id" >> dataset/reidentification.csv
        fi
    done
done
```

#### For Verification (`verification.csv`)

Format: `filename1,filename2,label` (label: 1=same identity, 0=different)

```bash
# Create pairs for verification (example script)
echo "filename1,filename2,label" > dataset/verification.csv
# Add pairs of images - 1 for same ID, 0 for different
# Example:
# after_4_bis/0/0.0.jpg,after_4_bis/0/0.1.jpg,1
# after_4_bis/0/0.0.jpg,after_4_bis/1/1.0.jpg,0
```

#### For Training (`train.csv`)

Same format as reidentification.csv: `filename,label`

---

## 3. Download Pretrained Models

Download pretrained models from the official Google Drive link (see README.md) and place them in the `pretrained/` folder:

- `pretrained/dog.pt` - Dog-specific model (194MB)
- `pretrained/unified.pt` - Multi-species model (457MB)

---

## 4. Running Inference

### Re-identification

Find matching identities across your dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/reidentification.py \
  -m arcface \
  -w pretrained/dog.pt \
  -i dataset/reidentification.csv \
  -o results/reidentification_results.csv \
  -b dataset
```

**Arguments:**
- `-m`: Method (`arcface`, `softmax`, `center`)
- `-w`: Path to pretrained weights
- `-i`: Input CSV file
- `-o`: Output results CSV
- `-b`: Base directory for images

**Compute Top-k Accuracy:**

```bash
python3 src/compute_topk_acc.py --topk 5 -i results/reidentification_results.csv
```

### Verification

Compare pairs of images to determine if they are the same identity:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/verification.py \
  -w pretrained/dog.pt \
  -i dataset/verification.csv \
  -o results/verification_results.csv \
  -b dataset \
  --network r50
```

**Compute AUC:**

```bash
python3 src/compute_auc.py -i results/verification_results.csv
```

---

## 5. Training on Your Custom Data

### Step 1: Create a Custom Config

Create `src/configs/custom.py`:

```python
from easydict import EasyDict as edict

config = edict()
config.network = "r50"           # Backbone: r50, ir50, swinb, vitb
config.embedding_size = 512      # Feature dimension
config.batch_size = 64           # Adjust based on GPU memory
config.num_epoch = 20            # Number of epochs
config.num_workers = 4
config.lr = 0.1
config.optimizer = "sgd"         # sgd or adamw
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace margins
config.fp16 = True               # Mixed precision training

# Data paths
config.basedir = 'dataset'
config.train_csv = 'dataset/train.csv'

# Number of unique identities in your dataset
config.num_classes = 1426        # Adjust to your actual number
```

### Step 2: Run Training

**ArcFace Training (Recommended):**

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train_arcface.py \
  src/configs/custom.py \
  --output outputs/custom/arcface
```

**Other Training Methods:**

```bash
# Softmax loss
python3 src/train_softmax.py src/configs/custom.py --output outputs/custom/softmax

# Center loss
python3 src/train_center.py src/configs/custom.py --output outputs/custom/center

# Triplet loss
python3 src/train_triplet.py src/configs/custom.py --output outputs/custom/triplet
```

### Step 3: Evaluate Your Trained Model

```bash
python3 src/reidentification.py \
  -m arcface \
  -w outputs/custom/arcface/model.pt \
  -i dataset/reidentification.csv \
  -o results/custom_results.csv \
  -b dataset
```

---

## 6. Face Alignment (Optional)

If your images are not pre-aligned, use the face alignment script:

### Step 1: Detect Keypoints

Use [AnyFace](https://github.com/IS2AI/AnyFace) to detect 5 facial keypoints and save them as `.npy` files.

### Step 2: Align Images

```bash
python3 src/face_align.py \
  --tgt /path/to/detected_keypoints.npy \
  --img /path/to/input_image.jpg \
  --src keypoints/dog.npy \
  --out /path/to/aligned_output.jpg
```

Reference keypoints are available in `keypoints/` for various species:
- `dog.npy`, `cat.npy`, `pig.npy`, `rabbit.npy`, etc.

---

## 7. Available Backbones

| Network | Description |
|---------|-------------|
| `r50`   | ResNet-50 (default) |
| `r101`  | ResNet-101 |
| `ir18`  | IResNet-18 |
| `ir50`  | IResNet-50 |
| `ir100` | IResNet-100 |
| `swinb` | Swin Transformer |
| `vitb`  | Vision Transformer |

---

## 8. Quick Start Example

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Generate CSV for your data
echo "filename,label" > dataset/reidentification.csv
for dir in dataset/after_4_bis/*/; do
    id=$(basename "$dir")
    for img in "$dir"*.jpg; do
        [ -f "$img" ] && echo "${img#dataset/},$id" >> dataset/reidentification.csv
    done
done

# 3. Create results directory
mkdir -p results

# 4. Run re-identification with pretrained model
CUDA_VISIBLE_DEVICES=0 python3 src/reidentification.py \
  -m arcface \
  -w pretrained/dog.pt \
  -i dataset/reidentification.csv \
  -o results/reid_results.csv \
  -b dataset

# 5. Compute accuracy
python3 src/compute_topk_acc.py --topk 5 -i results/reid_results.csv
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in your config
- Use a smaller backbone (e.g., `ir18` instead of `r50`)

### Missing Pretrained Models
- Download from the Google Drive link in README.md
- Place in `pretrained/` directory

### CSV Format Issues
- Ensure no trailing whitespace
- Use forward slashes in paths
- Labels should be integers

---

## References

- Paper: [PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](https://arxiv.org/abs/2407.13555)
- Official Repository: Check README.md for download links and citations

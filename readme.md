# MacroHFT
This is the official implementation of the KDD 2024 "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading".
https://arxiv.org/abs/2406.14537

## Environment Setup

### Conda Environment
Create a conda environment with Python 3.9:

```bash
conda create -n macrohft python=3.9
conda activate macrohft
```

### Install PyTorch
Please install the appropriate PyTorch version for your GPU configuration. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to select the version that suits your system.

Example installation commands:

**CPU version:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8 version:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.4 version (original):**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.6 version:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### Install Other Requirements
Install other necessary packages:
```bash
pip install -r requirements.txt
```

**Note:** If you install a different version of PyTorch, the PyTorch-related packages in requirements.txt may be overridden, which is normal.

## Running the Demo

### Data Preparation

1. Download the dataset from Google Drive:

   https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

2. Create the data/ETHUSDT directory if it doesn't exist:

   ```bash
   mkdir -p data/ETHUSDT
   ```

3. Extract and place all downloaded files inside the data/ETHUSDT directory. The final structure should look like:

   ```
   data/
     └── ETHUSDT/
         ├── df_test.feather
         ├── df_train.feather
         ├── df_val.feather
         └── ...
   ```

## Running the Pipeline

First, make all the shell scripts executable:

```bash
chmod +x scripts/decomposition.sh
chmod +x scripts/low_level.sh
chmod +x scripts/high_level.sh
```

### Step 1
Run the decomposition script for data decomposition and labeling:

```bash
./scripts/decomposition.sh
```

### Step 2
Run the low-level policy optimization script:

```bash
./scripts/low_level.sh
```

Update: We now provide trained model checkpoints for sub-agents, which can be directly used to train meta-policy.

### Step 3
Run the meta-policy optimization script:

```bash
./scripts/high_level.sh
``` 

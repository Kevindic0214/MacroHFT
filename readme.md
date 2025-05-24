# MacroHFT
This is the official implementation of the KDD 2024 "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading".
https://arxiv.org/abs/2406.14537

## Environment Setup

### Conda Environment
Create a conda environment with Python 3.9:

```bash
conda create -n macroHFT python=3.9
conda activate macroHFT
```

### Install Requirements
```bash
pip install -r requirements.txt
```

## Running the Demo

You may first download the dataset from Google Drive:

https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

and put the folder under data folder.

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

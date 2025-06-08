import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse

def load_and_describe_numpy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Get basic information
        info = {
            'file_name': os.path.basename(file_path),
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size,
            'bytes': data.nbytes,
        }
        
        # Get sample data (first few elements)
        if data.size > 0:
            if data.ndim == 0:  # scalar
                info['sample'] = str(data.item())
            elif data.ndim == 1:  # 1D array
                info['sample'] = str(data[:min(5, data.size)])
            else:  # multi-dimensional array
                flat_idx = min(5, data.size)
                info['sample'] = str(data.flatten()[:flat_idx])
                
            # Get statistics for numerical data
            if np.issubdtype(data.dtype, np.number):
                info['min'] = float(np.min(data))
                info['max'] = float(np.max(data))
                info['mean'] = float(np.mean(data))
                info['std'] = float(np.std(data))
                
        return info, data
    except Exception as e:
        return {'error': str(e)}, None

def print_file_info(info):
    print(f"\n{'='*50}")
    print(f"File: {info['file_name']}")
    print(f"{'='*50}")
    
    # Remove the file_name from the info to avoid duplication
    display_info = {k: v for k, v in info.items() if k != 'file_name'}
    
    # Convert info dict to a list of [key, value] for tabulate
    table = [[k, v] for k, v in display_info.items()]
    print(tabulate(table, headers=['Property', 'Value'], tablefmt='grid'))

def plot_data(data, file_name):
    if data is None or not np.issubdtype(data.dtype, np.number):
        return
    
    plt.figure(figsize=(10, 6))
    
    if data.ndim == 1:
        plt.plot(data)
        plt.title(f"{file_name} - Values")
        plt.xlabel("Index")
        plt.ylabel("Value")
    elif data.ndim == 2:
        if data.shape[1] <= 10:  # If not too many columns
            for i in range(data.shape[1]):
                plt.plot(data[:, i], label=f"Column {i}")
            plt.legend()
        else:
            plt.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar()
        plt.title(f"{file_name} - 2D Data")
    
    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(file_name)[0]}_plot.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Explore NumPy files in macrohft_result directory')
    parser.add_argument('--dir', type=str, default='macrohft_result', help='Directory containing NumPy files')
    parser.add_argument('--plot', action='store_true', help='Generate plots for the data')
    parser.add_argument('--file', type=str, help='Specific file to analyze (optional)')
    
    args = parser.parse_args()
    
    # Get the absolute path of the directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, args.dir)
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return
    
    print(f"Exploring NumPy files in: {target_dir}")
    
    # Get all .npy files in the directory
    if args.file:
        npy_files = [os.path.join(target_dir, args.file)]
        if not os.path.exists(npy_files[0]):
            print(f"Error: File '{npy_files[0]}' does not exist.")
            return
    else:
        npy_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                    if f.endswith('.npy')]
    
    if not npy_files:
        print("No .npy files found in the directory.")
        return
    
    print(f"Found {len(npy_files)} NumPy files.")
    
    # Process each file
    for file_path in npy_files:
        info, data = load_and_describe_numpy_file(file_path)
        
        if 'error' in info:
            print(f"\nError processing {os.path.basename(file_path)}: {info['error']}")
            continue
        
        print_file_info(info)
        
        if args.plot and data is not None:
            try:
                plot_data(data, os.path.basename(file_path))
                print(f"Plot saved as {os.path.splitext(os.path.basename(file_path))[0]}_plot.png")
            except Exception as e:
                print(f"Error creating plot: {e}")

if __name__ == "__main__":
    main()

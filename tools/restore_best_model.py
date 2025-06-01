import os
import numpy as np
import shutil
import argparse
import glob

def find_best_epoch_and_restore(base_seed_path):
    """
    Finds the best epoch based on validation return rates and copies its
    trained_model.pkl as best_model.pkl to the base_seed_path.

    Args:
        base_seed_path (str): The path to the directory containing epoch subdirectories
                              (e.g., ./result/low_level/ETHUSDT/slope.C51/0.0/label_1/seed_12345).
    """
    best_avg_return_rate = -float('inf')
    best_epoch_dir_name = None
    best_epoch_num = -1

    epoch_dirs = glob.glob(os.path.join(base_seed_path, "epoch_*"))
    if not epoch_dirs:
        print(f"錯誤：在 '{base_seed_path}' 中找不到任何 'epoch_*' 目錄。")
        return

    print(f"正在掃描 '{base_seed_path}' 中的 epoch 目錄...")

    for epoch_dir in epoch_dirs:
        if not os.path.isdir(epoch_dir):
            continue

        try:
            epoch_num_str = os.path.basename(epoch_dir).split('_')[-1]
            epoch_num = int(epoch_num_str)
        except ValueError:
            print(f"警告：無法從 '{os.path.basename(epoch_dir)}' 解析 epoch 編號，跳過。")
            continue
            
        val_path = os.path.join(epoch_dir, "val")
        if not os.path.isdir(val_path):
            print(f"警告：在 '{epoch_dir}' 中找不到 'val' 目錄，跳過 epoch {epoch_num}。")
            continue

        try:
            return_rate_0_path = os.path.join(val_path, "return_rate_mean_val_0.npy")
            return_rate_1_path = os.path.join(val_path, "return_rate_mean_val_1.npy")

            if not os.path.exists(return_rate_0_path):
                print(f"警告：在 '{val_path}' 中找不到 'return_rate_mean_val_0.npy'，跳過 epoch {epoch_num}。")
                continue
            if not os.path.exists(return_rate_1_path):
                print(f"警告：在 '{val_path}' 中找不到 'return_rate_mean_val_1.npy'，跳過 epoch {epoch_num}。")
                continue
                
            return_rate_0 = np.load(return_rate_0_path)
            return_rate_1 = np.load(return_rate_1_path)
            
            # 確保載入的是純量值
            if return_rate_0.ndim > 0: return_rate_0 = return_rate_0.item()
            if return_rate_1.ndim > 0: return_rate_1 = return_rate_1.item()

            avg_return_rate = (return_rate_0 + return_rate_1) / 2
            print(f"Epoch {epoch_num}: 平均驗證回報率 = {avg_return_rate:.6f} (Val0: {return_rate_0:.6f}, Val1: {return_rate_1:.6f})")

            if avg_return_rate > best_avg_return_rate:
                best_avg_return_rate = avg_return_rate
                best_epoch_dir_name = os.path.basename(epoch_dir)
                best_epoch_num = epoch_num

        except FileNotFoundError as e:
            print(f"警告：讀取 epoch {epoch_num} 的驗證回報率時發生錯誤：{e}，跳過。")
        except Exception as e:
            print(f"警告：處理 epoch {epoch_num} 時發生未知錯誤：{e}，跳過。")


    if best_epoch_dir_name:
        print(f"\n找到最佳 Epoch: {best_epoch_num}，其平均驗證回報率為: {best_avg_return_rate:.6f}")
        
        source_model_path = os.path.join(base_seed_path, best_epoch_dir_name, "trained_model.pkl")
        destination_model_path = os.path.join(base_seed_path, "best_model.pkl")

        if os.path.exists(source_model_path):
            try:
                shutil.copyfile(source_model_path, destination_model_path)
                print(f"成功將 '{source_model_path}' 複製到 '{destination_model_path}'")
            except Exception as e:
                print(f"錯誤：複製檔案時出錯：{e}")
        else:
            print(f"錯誤：在最佳 epoch 目錄 '{os.path.join(base_seed_path, best_epoch_dir_name)}' 中找不到 'trained_model.pkl'。")
    else:
        print("\n在指定的路徑下沒有找到有效的 epoch 資料來確定最佳模型。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根據 epoch 驗證結果恢復最佳模型。")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="包含 epoch 子目錄的 `seed_*` 目錄的路徑。\n"
             "例如：./result/low_level/ETHUSDT/slope.C51/0.0/label_1/seed_12345"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"錯誤：提供的路徑 '{args.path}' 不是一個有效的目錄。")
    else:
        find_best_epoch_and_restore(args.path) 
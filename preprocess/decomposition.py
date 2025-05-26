import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression

def smooth_data(data):
    print("平滑數據處理中...")
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, data)

def get_slope(smoothed_data):
    X = np.arange(len(smoothed_data)).reshape(-1, 1)
    model = LinearRegression().fit(X, smoothed_data)
    return model.coef_[0]

def get_slope_window(window):
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    y = filtfilt(b, a, window.values)
    X = np.arange(len(y)).reshape(-1, 1)   
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def chunk(df_train, df_val, df_test):
    print("開始進行數據分塊...")
    chunk_size = 4320
    chunks_created = 0
    for i in range(int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_train):
            df_chunk = df_train[start:end].reset_index(drop=True)
            df_chunk.to_feather('./data/ETHUSDT/train/df_{}.feather'.format(i))
            chunks_created += 1
    print(f"訓練數據分塊完成，共創建 {chunks_created} 個分塊")

    chunks_created = 0
    for i in range(int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_val):
            df_chunk = df_val[start:end].reset_index(drop=True)
            df_chunk.to_feather('./data/ETHUSDT/val/df_{}.feather'.format(i))
            chunks_created += 1
    print(f"驗證數據分塊完成，共創建 {chunks_created} 個分塊")

    chunks_created = 0
    for i in range(int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_test):
            df_chunk = df_test[start:end].reset_index(drop=True)
            df_chunk.to_feather('./data/ETHUSDT/test/df_{}.feather'.format(i))
            chunks_created += 1
    print(f"測試數據分塊完成，共創建 {chunks_created} 個分塊")

def label_slope(df_train, df_val, df_test):
    print("開始進行斜率標記...")
    chunk_size = 4320
    slopes_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_train):
            chunk = df_train['close'][start:end].values
            smoothed_chunk = smooth_data(chunk)
            slope = get_slope(smoothed_chunk)
            slopes_train.append(slope)
    print(f"訓練數據斜率計算完成，共 {len(slopes_train)} 個斜率值")

    slopes_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_val):
            chunk = df_val['close'][start:end].values
            smoothed_chunk = smooth_data(chunk)
            slope = get_slope(smoothed_chunk)
            slopes_val.append(slope)
    print(f"驗證數據斜率計算完成，共 {len(slopes_val)} 個斜率值")

    slopes_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_test):
            chunk = df_test['close'][start:end].values
            smoothed_chunk = smooth_data(chunk)
            slope = get_slope(smoothed_chunk)
            slopes_test.append(slope)
    print(f"測試數據斜率計算完成，共 {len(slopes_test)} 個斜率值")

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)
    print(f"斜率分箱邊界值：{bins}")

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    print(f"訓練數據各類別樣本數：{[len(indices) for indices in train_indices]}")
    with open('./data/ETHUSDT/train/slope_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = -100
    bins[-1] = 100
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]

    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    print(f"驗證數據各類別樣本數：{[len(indices) for indices in val_indices]}")
    with open('./data/ETHUSDT/val/slope_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    print(f"測試數據各類別樣本數：{[len(indices) for indices in test_indices]}")
    with open('./data/ETHUSDT/test/slope_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)
    print("斜率標記完成")

def label_volatility(df_train, df_val, df_test):
    print("開始進行波動性標記...")
    chunk_size = 4320
    volatilities_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_train):
            chunk = df_train[start:end].copy()
            chunk['return'] = chunk['close'].pct_change().fillna(0)
            volatility = chunk['return'].std()
            volatilities_train.append(volatility)
    print(f"訓練數據波動性計算完成，共 {len(volatilities_train)} 個波動性值")

    volatilities_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_val):
            chunk = df_val[start:end].copy()
            chunk['return'] = chunk['close'].pct_change().fillna(0)
            volatility = chunk['return'].std()
            volatilities_val.append(volatility)
    print(f"驗證數據波動性計算完成，共 {len(volatilities_val)} 個波動性值")
    
    volatilities_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end <= len(df_test):
            chunk = df_test[start:end].copy()
            chunk['return'] = chunk['close'].pct_change().fillna(0)
            volatility = chunk['return'].std()
            volatilities_test.append(volatility)
    print(f"測試數據波動性計算完成，共 {len(volatilities_test)} 個波動性值")

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)
    print(f"波動性分箱邊界值：{bins}")

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    print(f"訓練數據波動性各類別樣本數：{[len(indices) for indices in train_indices]}")
    with open('./data/ETHUSDT/train/vol_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = 0
    bins[-1] = 1
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]

    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    print(f"驗證數據波動性各類別樣本數：{[len(indices) for indices in val_indices]}")
    with open('./data/ETHUSDT/val/vol_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    print(f"測試數據波動性各類別樣本數：{[len(indices) for indices in test_indices]}")
    with open('./data/ETHUSDT/test/vol_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)
    print("波動性標記完成")

def label_whole(df):
    print("開始處理整體標記...")
    df = df.copy()
    window_size_list = [360]
    for i in range(len(window_size_list)):
        window_size = window_size_list[i]
        print(f"計算視窗大小為 {window_size} 的斜率特徵")
        df['slope_{}'.format(window_size)] = df['close'].rolling(window=window_size).apply(get_slope_window)
        df['return'] = df['close'].pct_change().fillna(0)
        print(f"計算視窗大小為 {window_size} 的波動性特徵")
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()
    print(f"整體標記完成，數據形狀：{df.shape}")
    return df

if __name__ == "__main__":
    print("程序開始執行...")
    os.makedirs('./data/ETHUSDT', exist_ok=True)
    print("正在讀取數據...")
    
    df_train = pd.read_feather('./data/ETHUSDT/df_train.feather')
    df_val = pd.read_feather('./data/ETHUSDT/df_val.feather')
    df_test = pd.read_feather('./data/ETHUSDT/df_test.feather')
    print(f"讀取完成。訓練集大小: {df_train.shape}, 驗證集大小: {df_val.shape}, 測試集大小: {df_test.shape}")

    os.makedirs('./data/ETHUSDT/train', exist_ok=True)
    os.makedirs('./data/ETHUSDT/val', exist_ok=True)
    os.makedirs('./data/ETHUSDT/test', exist_ok=True)
    os.makedirs('./data/ETHUSDT/whole', exist_ok=True)
    print("已創建所需目錄")

    chunk(df_train, df_val, df_test)
    label_slope(df_train, df_val, df_test)
    label_volatility(df_train, df_val, df_test)

    print("處理整體標記...")
    df_train = label_whole(df_train).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_val = label_whole(df_val).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_test = label_whole(df_test).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    print(f"處理後數據大小 - 訓練集: {df_train.shape}, 驗證集: {df_val.shape}, 測試集: {df_test.shape}")

    print("保存處理後的數據...")
    df_train.to_feather('./data/ETHUSDT/whole/train.feather')
    df_val.to_feather('./data/ETHUSDT/whole/val.feather')
    df_test.to_feather('./data/ETHUSDT/whole/test.feather')
    print("程序執行完成！")
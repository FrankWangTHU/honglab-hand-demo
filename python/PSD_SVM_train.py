# task3_train_svm_model.py
# =============================================================================
# Xu's BCI Research Assistant - Task 3: Multi-Gesture Classification
#
# 版本 v1.0: 训练PSD+SVM模型并保存以供在线测试
# =============================================================================

# %% 导入所需库
import mne
import numpy as np
import os
import joblib # 【新】用于保存和加载模型
import sys # 【新】导入sys库来处理路径问题

# --- 【核心修改】: 动态地将项目根目录(BCI)添加到Python的搜索路径中 ---
# 这能让Python解释器正确地找到 'dataloaders' 模块
# '..' 代表上一级目录，从 'machinehand' 到 'BCI'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"项目根目录 '{project_root}' 已添加到系统路径。")

# 导入机器学习库
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 导入你的自定义数据加载和预处理函数
from dataloaders.loader import load_neuracle
from dataloaders import neo

# %% ====================================================================
#  第一部分: 参数设置与数据加载
# =====================================================================
print("--- 步骤1: 定义参数并为分类任务提取Flex阶段数据 ---")

# --- 将路径改为相对于项目根目录的相对路径 ---
root_path = os.path.join(project_root, 'data', 'hand-gesture')
print(f"正在从以下路径加载数据: {root_path}")
gesture_event_map = {
    'ROS-handA': {'scissor': 1, 'six': 4, 'grasp': 7},
    'ROS-handB': {'index': 1, 'seven': 4, 'thumb': 7}
}
gesture_names = sorted(list(set(gesture_event_map['ROS-handA'].keys()) | set(gesture_event_map['ROS-handB'].keys())))
gesture_to_label = {name: i for i, name in enumerate(gesture_names)}
event_id_mapping = {str(i): i for i in range(20)}

tmin_flex, tmax_flex = 0, 3.0
reject_threshold = 400e-6

# --- 数据加载与分段 ---
all_epochs_list = []
all_labels_list = []

try:
    all_subfolders = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))])
except FileNotFoundError:
    print(f"错误：找不到数据路径 '{root_path}'。")
    all_subfolders = []

for folder_name in all_subfolders:
    print(f"  > 正在处理文件夹: {folder_name}")
    map_key = 'ROS-handA' if 'ROS-handA' in folder_name else 'ROS-handB' if 'ROS-handB' in folder_name else None
    if map_key is None: continue

    try:
        data_path = os.path.join(root_path, folder_name)
        raw = load_neuracle(data_path, 'ecog'); raw = neo.preprocessing(raw, reref_method='average')
        events, _ = mne.events_from_annotations(raw, event_id=event_id_mapping, verbose=False)
        
        for gesture_name, event_marker in gesture_event_map[map_key].items():
            event_id = {gesture_name: event_marker}
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin_flex, tmax=tmax_flex, preload=True, reject={'ecog': reject_threshold}, baseline=None, verbose=False)
            if len(epochs) > 0:
                all_epochs_list.append(epochs)
                all_labels_list.extend([gesture_to_label[gesture_name]] * len(epochs))
    except Exception as e:
        print(f"处理文件夹 {folder_name} 时发生错误: {e}")

if not all_epochs_list:
    print("错误：未能成功提取任何手势的Epochs数据。")
else:
    all_epochs = mne.concatenate_epochs(all_epochs_list, verbose=False)
    labels = np.array(all_labels_list)
    print(f"\n--- 数据准备完毕，共提取 {len(all_epochs)} 个trials ---")

# %% ====================================================================
#  第二部分: 特征提取 (使用【基线校准后】的PSD)与数据集划分
# =====================================================================


# --- 【核心修改】: 不要直接分割MNE的Epochs对象，而是分割其索引（ train_test_split只会将mne.epochs对象当作普通的数组或列表来处理） ---

# 1. 创建一个与 all_epochs 等长的索引数组
indices = np.arange(len(all_epochs))

# 2. 使用 train_test_split 分割索引，同时保持标签的分层抽样
train_indices, test_indices, y_train, y_test = train_test_split(
    indices, labels, test_size=0.25, random_state=42, stratify=labels
)

# 3. 使用分割后的索引从原始 all_epochs 对象中创建训练集和测试集
#    这样做可以确保 epochs_train 和 epochs_test 仍然是 mne.Epochs 对象
epochs_train = all_epochs[train_indices]
epochs_test = all_epochs[test_indices]


epochs_train_baseline = epochs_train.copy().crop(tmin=0.0, tmax=1.5, verbose=False)
epochs_train_flex = epochs_train.copy().crop(tmin=1.5, tmax=3.0, verbose=False)

psd_train_baseline_data = epochs_train_baseline.compute_psd(method='multitaper', fmin=5, fmax=150, n_jobs=-1, verbose=False).get_data()
#!psd_train_baseline_data维度：(n_trials, n_channels, n_freqs)
mean_psd_train_baseline = psd_train_baseline_data.mean(axis=0)
#!mean_psd_train_baseline维度：(n_channels, n_freqs)；之后在校准计算时会自动扩维，保证逐试次计算

psd_train_flex_data, freqs = epochs_train_flex.compute_psd(method='multitaper', fmin=5, fmax=150, n_jobs=-1, verbose=False).get_data(return_freqs=True)
#!psd_train_flex_data维度：(n_trials, n_channels, n_freqs)；freqs为一维数组(n_freqs,)显示每个频率点的值
psd_corrected_train = (psd_train_flex_data - mean_psd_train_baseline) / mean_psd_train_baseline
#!psd_corrected_train维度: (n_trials, n_channels, n_freqs)
n_trials_train, n_channels, n_freqs = psd_corrected_train.shape
X_train = psd_corrected_train.reshape(n_trials_train, n_channels * n_freqs)

epochs_test_flex = epochs_test.copy().crop(tmin=1.5, tmax=3.0, verbose=False)
psd_test_flex_data = epochs_test_flex.compute_psd(method='multitaper', fmin=5, fmax=150, n_jobs=-1, verbose=False).get_data()
psd_corrected_test = (psd_test_flex_data - mean_psd_train_baseline) / mean_psd_train_baseline
n_trials_test, _, _ = psd_corrected_test.shape
X_test = psd_corrected_test.reshape(n_trials_test, n_channels * n_freqs)

print(f"特征提取完毕，每个样本的特征维度为: {n_channels * n_freqs}")
print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
   
   # %% ====================================================================
#  第三部分: 模型训练与保存
# =====================================================================
print("\n--- 步骤3: 训练SVM模型 ---")
    
pipeline = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, kernel='rbf', class_weight='balanced')
)
    
print("正在训练SVM模型...")
pipeline.fit(X_train, y_train)
print("模型训练完成！")


# --- 【核心修改】: 将输出路径明确地设置为相对于当前脚本目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__)) #精确地获取当前脚本文件所在的文件夹路径（即.../BCI/machinehand）
output_dir = os.path.join(script_dir, 'model_data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, 'svm_classifier.joblib')
test_data_path = os.path.join(output_dir, 'test_data.npz')

print(f"\n正在保存已训练的模型至: {model_path}")
joblib.dump(pipeline, model_path)

print(f"正在保存测试数据以供后续使用至: {test_data_path}")
np.savez(test_data_path, X_test=X_test, y_test=y_test, gesture_names=gesture_names, n_channels=n_channels,  # <-- 新增
         n_freqs=n_freqs, freqs=freqs)

print("\n--- 训练脚本执行完毕 ---")

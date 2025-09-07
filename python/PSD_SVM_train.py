# task3_train_svm_model.py
# =============================================================================
# Xu's BCI Research Assistant - Task 3: Multi-Gesture Classification
#
# 版本 v4.0:
# - 【优化】采用更标准的ML流程：先合并所有特征，再进行一次性的分层抽样
# - 简化了数据集划分逻辑，不再需要对索引进行操作
# - 确保基线严格从划分后的训练集中提取
# =============================================================================

# %% 导入所需库
import mne
import numpy as np
import os
import joblib
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"项目根目录 '{project_root}' 已添加到系统路径。")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dataloaders.loader import load_neuracle
from dataloaders import neo

# %% ====================================================================
#  第一部分: 参数设置与数据加载 (与v3.0相同)
# =====================================================================
print("--- 步骤1: 定义参数并为分类任务提取数据 ---")

root_path = os.path.join(project_root, 'data', 'hand-gesture')
gesture_event_map = {
    'ROS-handA': {'scissor': 1, 'six': 4, 'grasp': 7},
    'ROS-handB': {'index': 1, 'seven': 4, 'thumb': 7}
}
gesture_names_only = sorted(list(set(gesture_event_map['ROS-handA'].keys()) | set(gesture_event_map['ROS-handB'].keys())))
gesture_names = gesture_names_only + ['Rest']
gesture_to_label = {name: i for i, name in enumerate(gesture_names)}
event_id_mapping = {str(i): i for i in range(20)}

tmin, tmax = 0, 3.0
reject_threshold = 400e-6

# --- 数据加载 ---
all_epochs_list = []
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
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True, reject={'ecog': reject_threshold}, baseline=None, verbose=False)
            if len(epochs) > 0:
                all_epochs_list.append(epochs)
    except Exception as e:
        print(f"处理文件夹 {folder_name} 时发生错误: {e}")

if not all_epochs_list:
    print("错误：未能成功提取任何手势的Epochs数据。")
    all_epochs_full = None
else:
    all_epochs_full = mne.concatenate_epochs(all_epochs_list, verbose=False)

# %% ====================================================================
#  第二部分: 特征提取 (基线校准) 与数据集划分 (重构)
# =====================================================================
print("\n--- 步骤2: 提取、合并特征，然后进行一次性分层抽样 ---")

# 1. 分别提取手势(Flex)和静息(Rest)的Epochs对象和标签
epochs_gestures = all_epochs_full.copy().crop(tmin=1.5, tmax=3.0)
labels_gestures = np.array([gesture_to_label[list(e.event_id.keys())[0]] for e in all_epochs_list for _ in range(len(e))])

epochs_rest = all_epochs_full.copy().crop(tmin=0, tmax=1.5)
labels_rest = np.full(len(epochs_rest), gesture_to_label['Rest'])

# 2. 分别计算它们的【原始】PSD值
psd_gestures_raw, freqs = epochs_gestures.compute_psd(method='multitaper', fmin=5, fmax=150, n_jobs=-1, verbose=False).get_data(return_freqs=True)
psd_rest_raw = epochs_rest.compute_psd(method='multitaper', fmin=5, fmax=150, n_jobs=-1, verbose=False).get_data()

# 3. 在特征层面（NumPy数组）将所有数据和标签合并
all_psd_raw_3d = np.concatenate([psd_gestures_raw, psd_rest_raw])
all_labels = np.concatenate([labels_gestures, labels_rest])

# 4. 【核心优化】对合并后的完整数据集进行一次性的分层抽样划分
X_train_raw_3d, X_test_raw_3d, y_train, y_test = train_test_split(
    all_psd_raw_3d, all_labels, test_size=0.25, random_state=42, stratify=all_labels
)

# 5. 【核心优化】严格地只从划分出的训练集中提取"Rest"样本来定义基线
rest_label_id = gesture_to_label['Rest']
psd_train_rest_samples = X_train_raw_3d[y_train == rest_label_id]
mean_psd_baseline = psd_train_rest_samples.mean(axis=0)

# 6. 使用这个基线来校准训练集和测试集
epsilon = 1e-10 # 防止除以零
X_train_corrected_3d = (X_train_raw_3d - mean_psd_baseline) / (mean_psd_baseline + epsilon)
X_test_corrected_3d = (X_test_raw_3d - mean_psd_baseline) / (mean_psd_baseline + epsilon)

# 7. 将校准后的特征压平以输入模型
n_trials_train, n_channels, n_freqs = X_train_corrected_3d.shape
X_train_flat = X_train_corrected_3d.reshape(n_trials_train, -1)
n_trials_test, _, _ = X_test_corrected_3d.shape
X_test_flat = X_test_corrected_3d.reshape(n_trials_test, -1)

print(f"特征提取完毕，训练集样本数: {len(X_train_flat)}, 测试集样本数: {len(X_test_flat)}")
   
# %% ====================================================================
#  第三部分: 模型训练与保存 (与v3.0相同)
# =====================================================================
print("\n--- 步骤3: 训练SVM模型 ---")
    
pipeline = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', class_weight='balanced'))
#!class_weight='balanced'通过加大对少数类样本的惩罚力度处理类别不平衡的数据

print("正在训练SVM模型...")
pipeline.fit(X_train_flat, y_train)
print("模型训练完成！")

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'model_data')
if not os.path.exists(output_dir): os.makedirs(output_dir)

model_path = os.path.join(output_dir, 'svm_classifier.joblib')
test_data_path = os.path.join(output_dir, 'test_data.npz')

print(f"\n正在保存已训练的模型至: {model_path}")
joblib.dump(pipeline, model_path)

print(f"正在保存测试数据以供后续使用至: {test_data_path}")
# 保存的数据结构和v3.0完全一样，因此test脚本无需修改
np.savez(test_data_path, 
         X_test_flat=X_test_flat,       # 校准后的2D特征，用于模型预测
         X_test_raw_3d=X_test_raw_3d,   # 未校准的3D原始PSD，用于可视化
         y_test=y_test, 
         gesture_names=gesture_names,
         freqs=freqs,
         mean_psd_baseline_for_viz=mean_psd_baseline # 用于可视化的基线
        )

print("\n--- 训练脚本执行完毕 ---")


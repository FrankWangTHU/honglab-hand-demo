# task3_test_online_simulation.py
# =============================================================================
# Xu's BCI Research Assistant - Task 3: Multi-Gesture Classification
#
# 版本 v3.0:
# - 模型预测使用【基线校准后】的PSD特征
# - 实时对【原始】可视化数据进行基线校准
# =============================================================================

# %% 导入所需库
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import math

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# %% ====================================================================
#  定义ITR计算函数
# =====================================================================
def calculate_itr(N, P, T):
    if P <= 1/N: return 0.0 # P=1/N时log(0)无意义，ITR为0
    if P == 1.0:
        bits_per_trial = math.log2(N)
    else:
        bits_per_trial = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    return bits_per_trial * (60 / T)

# %% ====================================================================
#  第一部分: 加载模型和测试数据 (修改)
# =====================================================================
print("--- 步骤1: 加载预训练模型和测试数据 ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model_data', 'svm_classifier.joblib')
test_data_path = os.path.join(script_dir, 'model_data', 'test_data.npz')

if not os.path.exists(model_path) or not os.path.exists(test_data_path):
    print("!!! 致命错误: 找不到模型文件或测试数据文件 !!!")
    print("请先确保你已经成功运行了更新版的'PSD_SVM_train.py'脚本。")
    import sys; sys.exit()

print(f"正在从 {model_path} 加载模型...")
model = joblib.load(model_path)
print("模型加载成功！")

print(f"正在从 {test_data_path} 加载测试数据...")
test_data = np.load(test_data_path, allow_pickle=True)
X_test_flat = test_data['X_test_flat']           # 【已校准】的2D特征，用于模型预测
X_test_raw_3d = test_data['X_test_raw_3d']       # 【未校准】的3D原始PSD，用于可视化
y_test = test_data['y_test']
gesture_names = test_data['gesture_names']
freqs = test_data['freqs']
mean_psd_baseline_for_viz = test_data['mean_psd_baseline_for_viz']

high_gamma_indices = np.where((freqs >= 50) & (freqs <= 100))[0]
if len(high_gamma_indices) == 0:
    print("!!! 警告: 未找到50-100Hz的频率点。")

print("测试数据加载成功！")
print(f"测试集样本数: {len(X_test_flat)}")
print(f"分类类别: {gesture_names}")


# %% ====================================================================
#  第二部分: 伪在线连续解码模拟
# =====================================================================
print("\n--- 步骤2: 开始伪在线解码模拟 ---")
print("模型预测使用【已校准】PSD, 可视化能量经实时校准。")
print("-" * 50)

y_pred_online = []

for i in range(len(X_test_flat)):
    # 使用【已校准】的特征进行模型预测
    current_features_flat = X_test_flat[i]
    predicted_label_index = model.predict(current_features_flat.reshape(1, -1))[0]
    #! predicted_label_index是预测出的标签索引，即数字0-6，对应关系为{'grasp': 0, 'index': 1, 'scissor': 2, 'seven': 3, 'six': 4, 'thumb': 5, 'Rest': 6}
    #! 这一数字可用于发送给机械手指令
    
    true_label_index = y_test[i]
    true_label_name = gesture_names[true_label_index]
    predicted_label_name = gesture_names[predicted_label_index]
    y_pred_online.append(predicted_label_index)
    
    # 对【未校准】的3D原始PSD进行实时基线校准，仅用于可视化
    current_sample_3d_raw = X_test_raw_3d[i]
    epsilon = 1e-10
    current_sample_3d_corrected = (current_sample_3d_raw - mean_psd_baseline_for_viz) / (mean_psd_baseline_for_viz + epsilon)
    
    # 从校准后的数据中计算可视化指标
    mean_psd_corrected = np.mean(current_sample_3d_corrected, axis=1)
    high_gamma_psd_corrected = current_sample_3d_corrected[:, high_gamma_indices]
    mean_high_gamma_power_corrected = np.mean(high_gamma_psd_corrected, axis=1)
    
    print(f"样本 {i+1}/{len(X_test_flat)} -> 预测: '{predicted_label_name}' (真实: '{true_label_name}')")
    
    print("  - Ch # | [校准后] High Gamma | [校准后] 总平均 PSD")
    print("  ----------------------------------------------------")
    for ch_idx, (hg_power, total_psd) in enumerate(zip(mean_high_gamma_power_corrected, mean_psd_corrected)):
        print(f"  - Ch {ch_idx+1:<2} | {hg_power:>20.3f} | {total_psd:>20.3f}")
    print("-" * 25)

    time.sleep(0.05)

print("-" * 50)
print("--- 伪在线解码模拟完成 ---")

# %% ====================================================================
#  第三部分: 完整性能评估
# =====================================================================
print("\n--- 步骤3: 进行最终的全面性能评估 ---")

y_pred = np.array(y_pred_online)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=gesture_names, zero_division=0)

print(f"\n----------- 评估结果 -----------")
print(f"总体分类准确率 (Overall Accuracy): {accuracy:.2%}\n")

N = len(gesture_names)
T = 1.5
itr = calculate_itr(N=N, P=accuracy, T=T)
print(f"信息传输率 (ITR): {itr:.2f} bits/min\n")

print("各手势详细评估报告 (Classification Report):")
print(report)
print(f"--------------------------------")

print("正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_names)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title('SVM 7-Class Confusion Matrix (Baseline Corrected PSD)') # 标题更新
plt.tight_layout(); plt.show()


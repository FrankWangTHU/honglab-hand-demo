# task3_test_online_simulation.py
# =============================================================================
# Xu's BCI Research Assistant - Task 3: Multi-Gesture Classification
#
# 版本 v1.0: 加载已训练的SVM模型并进行伪在线测试与评估
# =============================================================================

# %% 导入所需库
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib # 用于加载模型
import time
import math


# 导入机器学习库
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# %% ====================================================================
#  定义ITR计算函数 (与训练脚本中相同)
# =====================================================================
def calculate_itr(N, P, T):
    if P < 1/N: return 0.0
    if P == 1.0:
        bits_per_trial = math.log2(N)
    else:
        bits_per_trial = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    return bits_per_trial * (60 / T)

# %% ====================================================================
#  第一部分: 加载模型和测试数据
# =====================================================================
print("--- 步骤1: 加载预训练模型和测试数据 ---")

# --- 【核心修改】: 将加载路径明确地设置为相对于当前脚本目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__)) #精确地获取当前脚本文件所在的文件夹路径（即.../BCI/machinehand）
model_path = os.path.join(script_dir, 'model_data', 'svm_classifier.joblib')
test_data_path = os.path.join(script_dir, 'model_data', 'test_data.npz')

# 检查文件是否存在
if not os.path.exists(model_path) or not os.path.exists(test_data_path):
    print("!!! 致命错误: 找不到模型文件或测试数据文件 !!!")
    print("请先确保你已经成功运行了'PSD_SVM_train.py'脚本。")
    import sys; sys.exit()

# 加载模型和数据
print(f"正在从 {model_path} 加载模型...")
model = joblib.load(model_path)
print("模型加载成功！")

print(f"正在从 {test_data_path} 加载测试数据...")
test_data = np.load(test_data_path, allow_pickle=True)
X_test = test_data['X_test']
y_test = test_data['y_test']
gesture_names = test_data['gesture_names']
# --- 【核心修改 1】: 加载维度信息, 并将X_test恢复为3D ---
try:
    n_channels = test_data['n_channels']
    n_freqs = test_data['n_freqs']
    freqs = test_data['freqs'] # 加载频率轴
    # 将压平的特征数据恢复成 (n_trials, n_channels, n_freqs) 的三维形式
    X_test_3d = X_test.reshape(len(X_test), n_channels, n_freqs)
    print("成功加载维度信息并将特征数据恢复为3D。")
       
    # 找到50Hz到100Hz频率范围对应的索引
    high_gamma_indices = np.where((freqs >= 50) & (freqs <= 100))[0]
    
    if len(high_gamma_indices) == 0:
        print("!!! 警告: 在数据中未找到50-100Hz的频率点，无法计算High Gamma能量。 !!!")

    print("成功加载频率轴信息，并定位High Gamma频段。")

except KeyError:
    print("!!! 致命错误: 测试数据文件中找不到 'n_channels' 或 'n_freqs'。 !!!")
    print("请务必先运行更新版的 'PSD_SVM_train.py' 来重新生成测试数据文件。")
    sys.exit()
print("测试数据加载成功！")
print(f"测试集样本数: {len(X_test)}")

# %% ====================================================================
#  第二部分: 伪在线连续解码模拟
# =====================================================================
print("\n--- 步骤2: 开始伪在线解码模拟 ---")
print("将逐个解码测试集中的样本，并显示预测结果和真实标签。")
print("-" * 40)

# 用于存储所有预测结果
y_pred_online = []

# 遍历测试集中的每一个样本
for i in range(len(X_test)):
    # 获取当前时间窗口的特征
    current_features = X_test[i]
    # 获取真实标签
    true_label_index = y_test[i]
    true_label_name = gesture_names[true_label_index]
    
    # --- 核心解码步骤 ---
    # scikit-learn模型期望一个二维输入 (n_samples, n_features)
    # 所以我们需要将单个样本 reshape 成 (1, n_features)
    predicted_label_index = model.predict(current_features.reshape(1, -1))[0]
    predicted_label_name = gesture_names[predicted_label_index]
    
    # 将预测结果存起来
    y_pred_online.append(predicted_label_index)
    
    # --- 【核心修改 2】: 计算并输出每个通道的平均PSD ---
    # 获取当前样本的3D数据 (n_channels, n_freqs)
    current_sample_3d = X_test_3d[i]
    # 1. 沿着频率轴(axis=1)求均值，得到每个通道的平均PSD
    mean_psd = np.mean(current_sample_3d, axis=1)
    # 2. 计算High Gamma频段的平均能量
    high_gamma_psd = current_sample_3d[:, high_gamma_indices]
    mean_high_gamma_power = np.mean(high_gamma_psd, axis=1)
    
    # 打印实时解码结果
    print(f"样本 {i+1}/{len(X_test)} -> 预测结果: '{predicted_label_name}' (真实手势: '{true_label_name}')")
    
    # 使用zip同时遍历两个结果，并格式化打印
    print("  - Ch # | High Gamma 能量 | 总平均 PSD")
    print("  --------------------------------------")
    for ch_idx, (hg_power, total_psd) in enumerate(zip(mean_high_gamma_power, mean_psd)):
        print(f"  - Ch {ch_idx+1:<2} | {hg_power:>15.3f} | {total_psd:>12.3f}")
    print("-" * 25)

    # 增加一个小的延迟，让结果更容易观察
    time.sleep(0.05)

print("-" * 40)
print("--- 伪在线解码模拟完成 ---")

# %% ====================================================================
#  第三部分: 完整性能评估 (与原始脚本相同)
# =====================================================================
print("\n--- 步骤3: 进行最终的全面性能评估 ---")

y_pred = np.array(y_pred_online)

# --- 评估报告 ---
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=gesture_names)

print(f"\n----------- 评估结果 -----------")
print(f"总体分类准确率 (Overall Accuracy): {accuracy:.2%}\n")

# --- 计算并打印ITR ---
N = len(gesture_names)
T = 1.5 # Flex阶段时长为 3.0s - 1.5s = 1.5s
itr = calculate_itr(N=N, P=accuracy, T=T)
print(f"信息传输率 (ITR): {itr:.2f} bits/min\n")

print("各手势详细评估报告 (Classification Report):")
print(report)
print(f"--------------------------------")

# 可视化混淆矩阵
print("正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title('SVM Classification Confusion Matrix (PSD Features)')
plt.tight_layout(); plt.show()

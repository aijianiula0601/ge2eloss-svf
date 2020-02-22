import numpy as np

"""
功能：演示计算错误拒绝率和错误接受率的计算方式
"""

threshold_p = 0.5
print("# ------------------------------------------------------------------------")
print("# 错误拒绝率FRP")
print("# ------------------------------------------------------------------------")

np.random.seed(20)
utterances_num = 4

compare_total_num = (utterances_num - 1) * utterances_num / 2
print('compare_total_num:', compare_total_num)
intra_class_matrix = np.random.random((utterances_num, utterances_num))
print("-------------------------------")
print("类内比较矩阵:\n")
print(intra_class_matrix)

# 由于无法做到先把小于阀值的置为1，然后大于阀值置为0，所以先反过来
intra_class_matrix[intra_class_matrix >= threshold_p] = 1
intra_class_matrix[intra_class_matrix < threshold_p] = 0

# 把上三角置为0
intra_class_matrix = np.tri(utterances_num, utterances_num, k=-1) * intra_class_matrix
print("-------------------------------")
print("接受矩阵(置为1的正常):\n")
print(intra_class_matrix)

# 错误拒绝个数
mis_reject_num = compare_total_num - np.sum(intra_class_matrix)
print('mis_reject_num:', mis_reject_num)

# 错误拒绝率
fpr_value = mis_reject_num / compare_total_num

print("-------------------------------")
print("错误拒绝率为:", fpr_value)

print("# ------------------------------------------------------------------------")
print("# 错误接受率FAP")
print("# ------------------------------------------------------------------------")

speakers_num = 4
compare_speakers_num = (speakers_num - 1) * speakers_num / 2
print('compare_speakers_num:', compare_speakers_num)
np.random.seed(20)
inter_class_matrix = np.random.random((speakers_num, speakers_num))
print("-------------------------------")
print("类间比较矩阵:\n")
print(inter_class_matrix)

# 大于阀值的置为1，其他为0
inter_class_matrix[inter_class_matrix >= threshold_p] = 1
inter_class_matrix[inter_class_matrix < threshold_p] = 0

# 把上三角置为0
inter_class_matrix = np.tri(speakers_num, speakers_num, k=-1) * inter_class_matrix
print("-------------------------------")
print("接受矩阵(置为1的正常):\n")
print(inter_class_matrix)

# 错误个数
mis_accept_num = np.sum(inter_class_matrix)
print('mis_accept_num:', mis_accept_num)

# 错误接受率
far_value = mis_accept_num / compare_speakers_num
print("-------------------------------")
print("错误接受率为:", far_value)

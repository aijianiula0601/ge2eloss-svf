import numpy as np

"""
功能：演示计算错误拒绝率和错误接受率的计算方式
"""

np.random.seed(20)


def frp(intra_class_matrix, threshold_p):
    """
    计算错误拒绝率
    :param intra_class_matrix: 类内比较矩阵
    :param threshold_p: 阀值
    :return: 错误拒绝率
    """
    utterances_num = np.shape(intra_class_matrix)[0]
    compare_total_num = (utterances_num - 1) * utterances_num / 2

    # 由于无法做到先把小于阀值的置为1，然后大于阀值置为0，所以先反过来
    intra_class_matrix[intra_class_matrix >= threshold_p] = 1
    intra_class_matrix[intra_class_matrix < threshold_p] = 0

    # 把上三角置为0
    intra_class_matrix = np.tri(utterances_num, utterances_num, k=-1) * intra_class_matrix

    # 错误拒绝个数
    mis_reject_num = compare_total_num - np.sum(intra_class_matrix)

    # 错误拒绝率
    fpr_value = mis_reject_num / compare_total_num

    return fpr_value


def fap(inter_class_matrix, threshold_p):
    """
    计算错误接受率
    :param inter_class_matrix: 类间矩阵
    :param threshold_p: 阀值
    :return: 错误接受率
    """
    speakers_num = np.shape(inter_class_matrix)[0]
    compare_speakers_num = (speakers_num - 1) * speakers_num / 2

    # 大于阀值的置为1，其他为0
    inter_class_matrix[inter_class_matrix >= threshold_p] = 1
    inter_class_matrix[inter_class_matrix < threshold_p] = 0

    # 把上三角置为0
    inter_class_matrix = np.tri(speakers_num, speakers_num, k=-1) * inter_class_matrix

    # 错误个数
    mis_accept_num = np.sum(inter_class_matrix)

    # 错误接受率
    far_value = mis_accept_num / compare_speakers_num

    return far_value


if __name__ == '__main__':
    threshold_p = 0.5
    utterances_num = 4
    np.random.seed(20)
    intra_class_matrix = np.random.random((utterances_num, utterances_num))

    frp_value = frp(intra_class_matrix, threshold_p)
    print("错误拒绝率:", frp_value)

    speakers_num = 4
    np.random.seed(20)
    inter_class_matrix = np.random.random((speakers_num, speakers_num))
    fap_vlaue = fap(inter_class_matrix, threshold_p)
    print("错误接受率:", fap_vlaue)

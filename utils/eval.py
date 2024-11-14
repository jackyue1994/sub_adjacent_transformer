import numpy as np
import pandas as pd
import torch
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tadpak import pak
import time
import os
from scipy.stats import norm
import torch.nn.functional as F


def get_fp_tp_rate(predict, actual):
    tn, fp, fn, tp = confusion_matrix(actual, predict, labels=[0, 1]).ravel()

    # recall
    true_pos_rate = tp / (tp + fn)
    #
    false_pos_rate = fp / (fp + tn)

    return false_pos_rate, true_pos_rate


def pak_protocol(scores, labels, threshold, max_k=100):
    f1s = []
    ks = []
    fprs = []
    tprs = []
    preds = []

    for k in range(0, max_k + 1, 10):  # modify to range(0, max_k + 1, 1) for more precise result
        ks.append(k / 100)
        adjusted_preds = pak.pak(scores, labels, threshold, k=k)
        f1 = f1_score(labels, adjusted_preds)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        # print(f1)
        # print(k)
        f1s.append(f1)
        preds.append(adjusted_preds)

    area_under_f1 = auc(ks, f1s)
    max_f1_k = max(f1s)
    k_max = f1s.index(max_f1_k)
    preds_for_max = preds[f1s.index(max_f1_k)]
    # import matplotlib.pyplot as plt
    # plt.cla()
    # plt.plot(ks, f1s)
    # plt.savefig('DiffusionAE/plots/PAK_PROTOCOL')
    # print(f'AREA UNDER CURVE {area}')
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs


def evaluate(score, label, validation_thresh=None):
    if len(score) != len(label):
        score = score[:len(label)]
    false_pos_rates = []
    true_pos_rates = []
    f1s = []
    max_f1s_k = []
    preds = []
    # thresholds = np.arange(0, score.max(), min(0.001, score.max()/50))#0.001
    thresholds = np.arange(0, score.max() + 1, score.max() / 50)  # 0.001

    max_ks = []
    pairs = []

    for thresh in thresholds:
        f1, max_f1_k, k_max, best_preds, fprs, tprs = pak_protocol(score, label, thresh)
        max_f1s_k.append(max_f1_k)
        max_ks.append(k_max)
        preds.append(best_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        f1s.append(f1)
        pairs.extend([(thresh, i) for i in range(101)])

    if validation_thresh:
        print(f'validation_thresh is provided: {validation_thresh}')
        f1, max_f1_k, max_k, best_preds, _, _ = pak_protocol(score, label, validation_thresh)
    else:
        print('validation_thresh is not provided')
        f1 = max(f1s)
        max_possible_f1 = max(max_f1s_k)
        max_idx = max_f1s_k.index(max_possible_f1)
        max_k = max_ks[max_idx]
        thresh_max_f1 = thresholds[max_idx]
        best_preds = preds[max_idx]
        best_thresh = thresholds[f1s.index(f1)]

    roc_max = auc(np.transpose(np.array(false_pos_rates))[max_k], np.transpose(np.array(true_pos_rates))[max_k])
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score_pa.npy', np.transpose(false_pos_rates)[0])
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score_pa.npy', np.transpose(true_pos_rates)[0])

    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()

    sorted_indexes = np.argsort(false_pos_rates)
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    pairs = np.array(pairs)[sorted_indexes]
    roc_score = auc(false_pos_rates, true_pos_rates)

    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score.npy', true_pos_rates)
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score.npy', false_pos_rates)
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/pairs_diff_score.npy', pairs)
    # preds = predictions[f1s.index(f1)]
    if validation_thresh:
        return {
            'f1-AUC': f1,  # f1_k(area under f1) for validation threshold
            'ROC/AUC': roc_score,  # for all ks and all thresholds obtained on test scores
            'f1_max': max_f1_k,  # best f1 across k values
            'preds': best_preds,  # corresponding to best k
            'k': max_k,  # the k value correlated with the best f1 across k=1,100
            'thresh_max': validation_thresh,
            'roc_max': roc_score,
        }
    else:
        return {
            'f1-AUC': f1,
            'ROC/AUC': roc_score,
            'threshold': best_thresh,
            'f1_max': max_possible_f1,
            'roc_max': roc_max,
            'thresh_max': thresh_max_f1,
            'preds': best_preds,
            'k': max_k,
        }, false_pos_rates, true_pos_rates


def get_pred_from_loss(test_energy, test_rec_loss, thresh, thresh_rec_loss):
    pred = (test_energy > thresh).astype(int)
    pred2 = (test_rec_loss > thresh_rec_loss).astype(int)

    fixMode = False
    for i in range(len(pred)):
        if pred[i] == 1 and not fixMode:
            fixMode = True
            for j in range(i - 1, 0, -1):
                if pred2[j]:
                    pred[j] = 1
                else:
                    break
            for j in range(i + 1, len(pred)):
                if pred2[j]:
                    pred[j] = 1
                else:
                    break
        elif pred[i] == 0:
            fixMode = False

    return pred


def write_into_xls(att_matrix, excel_name):
    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    dataframe = pd.DataFrame(att_matrix)
    # print(dataframe)
    # print(excel_name)
    dataframe.to_excel(excel_name, index=False)


def myplot(test_labels, test_data, cri_loss, rec_loss, att_loss, dataset_name='dataset', anomaly_score=None):
    ind = np.nonzero(test_labels)[0]
    win_size = 1000

    nums = np.arange(test_data.shape[-1])
    dim = np.random.choice(nums, 2, replace=True)

    # print(ind)
    start = np.maximum(np.random.choice(ind) - 100, 0)
    x = np.arange(start, np.minimum(start + win_size, len(test_labels)))
    y1 = test_labels[x]
    y2 = test_data[x, dim[0]]
    y3 = test_data[x, dim[1]]

    print(f'testdata max: {np.max(test_data)}; min: {np.min(test_data)}')

    y4 = cri_loss[x]
    y5 = rec_loss[x]
    y6 = att_loss[x]

    # 创建一个2*2的子图布局
    fig, axs = plt.subplots(2, 3)

    # 在第一个子图上画第一条曲线
    axs[0, 0].plot(x, y1)
    axs[0, 0].set_title(dataset_name + ' test_labels')

    axs[0, 1].plot(x, y2)
    axs[0, 1].set_title(f'{dataset_name} data dim:{dim[0]}')

    axs[0, 2].plot(x, y3)
    axs[0, 2].set_title(f'{dataset_name} data dim:{dim[1]}')

    # 在第二个子图上画第二条曲线
    axs[1, 0].plot(x, y4)
    axs[1, 0].set_title('cri_loss')

    # 在第三个子图上画第三条曲线
    axs[1, 1].plot(x, y5)
    axs[1, 1].set_title('rec_loss')

    # 在第四个子图上画第四条曲线
    axs[1, 2].plot(x, y6)
    axs[1, 2].set_title('att_loss')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 保存图像为png格式，文件名为当前时间戳
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    os.makedirs('output', exist_ok=True)
    fig.savefig(os.path.join('output', f'{dataset_name}-{timestamp}.png'))

    # save to excel
    if len(test_data) < 1000:
        excel_name = os.path.join('output', f'{dataset_name}-{timestamp}-rec-attn-score-energy.xlsx')
        if anomaly_score is not None:
            att_matrix = np.concatenate((test_data, rec_loss.reshape(-1, 1), att_loss.reshape(-1, 1),
                                         anomaly_score.reshape(-1, 1), cri_loss.reshape(-1, 1),
                                         test_labels.reshape(-1, 1)), axis=1)
        else:
            att_matrix = np.concatenate((test_data, rec_loss.reshape(-1, 1), att_loss.reshape(-1, 1),
                                         cri_loss.reshape(-1, 1),  test_labels.reshape(-1, 1),), axis=1)
        write_into_xls(att_matrix, excel_name)


def compute_longest_anomaly(test_labels):
    list_ = []
    countFlag = False
    count = 0
    for label in test_labels:
        if label:
            if countFlag:
                count = count + 1
            else:
                countFlag = True
                count = 1
        else:
            if countFlag:
                countFlag = False
                list_.append(count)
                count = 0
    if count:
        list_.append(count)

    return np.max(np.array(list_)), np.min(np.array(list_))


def plot_mat(att_matrix, str0='tmp'):
    if not isinstance(att_matrix, np.ndarray):
        att_matrix = np.array(att_matrix)
    fig, axs = plt.subplots(1, 1)
    plt.imshow(att_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    plt.savefig(os.path.join('output', f'attn_mat_{str0}-{timestamp}.png'))
    plt.show()
    # save to excel
    excel_name = os.path.join('output', f'attn_mat_{str0}-{timestamp}.xlsx')
    write_into_xls(att_matrix, excel_name)


def myLoss(queries, keys, span=None, one_side=True):
    # queries,keys : B,L,H,D  --> output: B,L
    # how much the point can help others; anomaly helps little while the normals help more
    L = queries.shape[1]
    if span is None:
        span = [20, 30]

    assert L >= span[1] >= span[0] >= 0

    z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
    # lossMat0 = (queries * keys).sum(dim=-1) * z
    lossMat = None
    for k in range(-span[1], span[1] + 1):  # range(-span[1], -span[0]+1)
        # only one-side is used
        if one_side:
            if k < span[0]:
                continue
        else:
            if abs(k) < span[0]:
                continue

        shifted_queries = torch.roll(queries, shifts=k, dims=1)
        shifted_z = torch.roll(z, shifts=k, dims=1)

        if lossMat is None:
            # b,l,h
            lossMat = (shifted_queries * keys).sum(dim=-1) * shifted_z
        else:
            lossMat += (shifted_queries * keys).sum(dim=-1) * shifted_z

    # b,l,h
    lossMat = torch.mean(lossMat, dim=-1)

    return lossMat  # B,L


def myLossNew(queries, keys, span=None, one_side=True):
    # queries,keys : B,L,H,D  --> output: B,L
    # explicitly compute attention matrix
    # how much the point can help others; anomaly helps little while the normals help more
    L = queries.shape[1]
    if span is None:
        span = [20, 30]

    assert L >= span[1] >= span[0] >= 0

    # compute attention matrix
    attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", queries, keys)
    attnMatrix = attnMatrix / attnMatrix.sum(dim=-1, keepdim=True)

    lossMat = None
    for k in range(-span[1], span[1] + 1):  # range(-span[1], -span[0]+1)
        # only one-side is used
        if one_side:
            if k < span[0]:
                continue
        else:
            if abs(k) < span[0]:
                continue

        diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        if k > 0:
            p1d = (k, 0)
        else:
            p1d = (0, abs(k))
        diag1 = F.pad(diag1, p1d)

        if lossMat is None:
            lossMat = diag1
        else:
            lossMat += diag1

        if k > 0:
            offset_k = -(L-k)
        else:
            offset_k = L+k
        diag1 = torch.diagonal(attnMatrix, offset=offset_k, dim1=-2, dim2=-1)  # why use L-k ?  L-k performs better
        if offset_k > 0:
            p1d = (offset_k, 0)
        else:
            p1d = (0, abs(offset_k))
        diag1 = F.pad(diag1, p1d)

        lossMat += diag1

    # b,h,l
    lossMat = torch.mean(lossMat, dim=-2)

    return lossMat  # B,L


def myLoss2(attnMatrix, keys=None, span=None, one_side=True):
    # traditional self attention
    # b h l l

    B, H, L, _ = attnMatrix.shape
    lossMat = None
    for k in range(20, 30):
        diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        p1d = (k, 0)
        diag1 = F.pad(diag1, p1d)

        if lossMat is None:
            lossMat = diag1
        else:
            lossMat += diag1

        # -(L-k)
        diag1 = torch.diagonal(attnMatrix, offset=-(L-k), dim1=-2, dim2=-1)  # why use L-k ?  L-k performs better
        p1d = (0, L-k)  # p1d = (0, k)
        diag1 = F.pad(diag1, p1d)
        lossMat += diag1

        # # -k
        # diag1 = torch.diagonal(attnMatrix, offset=-k, dim1=-2, dim2=-1)
        # p1d = (0, k)
        # diag1 = F.pad(diag1, p1d)
        # lossMat += diag1
        #
        # # (L-k)
        # diag1 = torch.diagonal(attnMatrix, offset=L-k, dim1=-2, dim2=-1)
        # p1d = (L-k, 0)
        # diag1 = F.pad(diag1, p1d)
        # lossMat += diag1

        lossMat += diag1

    lossMat = torch.mean(lossMat, dim=1)

    return lossMat


def myLoss0(queries, keys):
    # all points are used, except itself
    attn_mat = queries.permute([0, 2, 1, 3]) @ keys.permute([0, 2, 3, 1])
    z = 1 / (torch.einsum("b l h e, b h e -> b h l", queries, keys.sum(dim=1)) + 1e-6)
    attn_mat = attn_mat * z.unsqueeze(-1)
    loss_mat = attn_mat.sum(dim=-2) - torch.diagonal(attn_mat, offset=0, dim1=-2, dim2=-1)
    return loss_mat.mean(dim=1)


def softmax(x, temperature=1, window=None):
    # softmax for numpy
    # print(x)
    x = x * temperature
    shape = x.shape[0]
    if window is not None:
        window = max(min(int(window), shape),1)
        rem = shape % window
        if rem != 0:
            x = np.concatenate([x, x[:int(window - rem)]], axis=0)

        x = x.reshape(-1, window)

    x = x.clip(-100, 100)
    output = (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)).reshape(-1)
    # print(output)
    return output[:shape]


def sliding_window_mean_std(vector, window_size=100):
    vector = previous_vector(vector, shift=5)
    mean = np.convolve(vector, np.ones(window_size), 'same') / window_size
    if len(mean) > len(vector):
        mean = mean[:len(vector)]
    std = np.sqrt(np.convolve(np.square(vector - mean), np.ones(window_size), 'same') / (window_size - 1))
    return mean, std


def previous_vector(vector, shift=10):
    return np.roll(vector, shift=-abs(shift))


def get_probs_from_cri(results, window_size=10):
    std_gauss = norm(loc=0, scale=1)
    mean, _ = sliding_window_mean_std(results, window_size=window_size)
    # _, std = sliding_window_mean_std(results, window_size=1000)
    std = results.std()
    # print(std)
    probs = -std_gauss.logsf((results - mean) / std)
    return probs


def use_smooth(vector, kernel_length=100):
    if kernel_length <= 0:
        return vector
    std_gauss = norm(loc=0, scale=1)
    x = np.linspace(-3, 3, int(kernel_length))
    kernel = std_gauss.pdf(x)
    vector = np.convolve(vector, kernel, 'same')
    return vector

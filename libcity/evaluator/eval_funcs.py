import numpy as np
import torch


# ????(Mean Square Error)
def mse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE: ??????????????'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


# ??????(Mean Absolute Error)
def mae(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE: ??????????????'
    return np.mean(sum(loc_pred - loc_true))


# ?????(Root Mean Square Error)
def rmse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE: ??????????????'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


# ?????????(Mean Absolute Percentage Error)
def mape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE: ??????????????'
    assert 0 not in loc_true, "MAPE: ?????0,??????"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


# ?????????(Mean Absolute Relative Error)
def mare(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE:??????????????"
    assert np.sum(loc_true) != 0, "MARE:??????0,??????"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


# ???????????(Symmetric Mean Absolute Percentage Error)
def smape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE: ??????????????'
    assert 0 in (loc_pred + loc_true), "SMAPE: ??????????0,??????"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) +
                                                        np.abs(loc_true)))


# ??????????????????
def acc(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy: ??????????????"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


def top_k(loc_pred, loc_true, topk):
    """
    count the hit numbers of loc_true in topK of loc_pred, used to calculate Precision, Recall and F1-score,
    calculate the reciprocal rank, used to calcualte MRR,
    calculate the sum of DCG@K of the batch, used to calculate NDCG

    Args:
        loc_pred: (batch_size * output_dim)
        loc_true: (batch_size * 1)
        topk:

    Returns:
        tuple: tuple contains:
            hit (int): the hit numbers \n
            rank (float): the sum of the reciprocal rank of input batch \n
            dcg (float): dcg
    """
    assert topk > 0, "top-k ACC????:k?????1"
    loc_pred = torch.FloatTensor(loc_pred)
    val, index = torch.topk(loc_pred, topk, 1)
    #print(loc_true)
    #print(index)
    #print('@'*3)
    index = index.numpy()
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg

def Precision_torch(preds, labels, topk):
    precision = []
    for i in range(preds.shape[0]):
        label = labels[i]
        pred = preds[i]
        accident_grids = label > 0
        sorted, _ = torch.sort(pred.flatten(), descending=True)
        threshold = sorted[topk - 1]
        pred_grids = pred >= threshold
        matched = pred_grids & accident_grids
        precision.append(torch.sum(matched.flatten()).item() / topk)
    return sum(precision) / len(precision)

def Recall_torch(preds, labels, topk):
    recall = []
    for i in range(preds.shape[0]):
        label = labels[i]
        pred = preds[i]
        accident_grids = label > 0
        sorted, _ = torch.sort(pred.flatten(), descending=True)
        threshold = sorted[topk - 1]
        pred_grids = pred >= threshold
        matched = pred_grids & accident_grids
        if torch.sum(accident_grids).item() != 0:
            recall.append(torch.sum(matched.flatten()).item() / torch.sum(accident_grids.flatten()).item())
    return sum(recall) / len(recall)

def F1_Score_torch(preds, labels, topk):
    precision = Precision_torch(preds, labels, topk)
    recall = Recall_torch(preds, labels, topk)
    return 2 * precision * recall / (precision + recall)



def MAP_torch(preds, labels, topk):
    ap = []
    for i in range(preds.shape[0]):
        label = labels[i].flatten()
        pred = preds[i].flatten()
        accident_grids = label > 0
        sorted, rank = torch.sort(pred, descending=True)
        rank = rank[:topk]
        if topk != 0:
            threshold = sorted[topk - 1]
        else:
            threshold = 0
        label = label != 0
        pred = pred >= threshold
        matched = pred & label
        match_num = 0
        precision_sum = 0
        for i in range(rank.shape[0]):
            if matched[rank[i]]:
                match_num += 1
                precision_sum += match_num / (i + 1)
        if rank.shape[0] != 0:
            ap.append(precision_sum / rank.shape[0])
    return sum(ap) / len(ap)


def PCC_torch(preds, labels, topk):
    pcc = []
    for i in range(preds.shape[0]):
        label = labels[i].flatten()
        pred = preds[i].flatten()
        sorted, rank = torch.sort(pred, descending=True)
        pred = sorted[:topk]
        rank = rank[:topk]
        sorted_label = torch.zeros(topk)
        for i in range(topk):
            sorted_label[i] = label[rank[i]]
        label = sorted_label
        label_average = torch.sum(label) / (label.shape[0])
        pred_average = torch.sum(pred) / (pred.shape[0])
        if torch.sqrt(torch.sum((label - label_average) * (label - label_average))) * torch.sqrt(
                torch.sum((pred - pred_average) * (pred - pred_average))) != 0:
            pcc.append((torch.sum((label - label_average) * (pred - pred_average)) / (
                    torch.sqrt(torch.sum((label - label_average) * (label - label_average))) * torch.sqrt(
                    torch.sum((pred - pred_average) * (pred - pred_average))))).item())
    return sum(pcc) / len(pcc)

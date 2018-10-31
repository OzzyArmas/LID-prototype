from collections import defaultdict

def get_recall(conf_matrix):
    recall = [conf_matrix[lang][lang] for lang in range(len(conf_matrix))]
    for idx, rec in enumerate(recall)):    
        recall[idx] = rec / sum(conf_matrix[idx])

    return recall

def get_precision(conf_matrix):
    precision = [conf_matrix[lang][lang] for lang in range(len(conf_matrix))]
    for idx, prec in enumerate(precision)):
        precision[idx] = prec / sum(conf_matrix.transpose()[idx])

    return precision


def get_fscore(precision, recall):
    # fscore = 2 * (precision * recall) / (precision + recall)
    fscore = {}
    for idx,pre,rec in enumerate(zip(precision, recall)):
        fscore[idx] = 2 * (prec * rec) / (prec + rec)
    return fscore

def get_error_rates(FA, FR, total):
    FAR = FA
    FRR = FR
    EER = defaultdict(float)
    for k1,k2 in zip(FAR, FRR):
        FAR[k1] /= total
        FRR[k2] /= total
        EER[k1] = (FAR[k1] + FRR[k2]) / 2

    return FAR, FRR, EER
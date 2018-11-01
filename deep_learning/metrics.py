from collections import defaultdict

def get_recall(conf_matrix):
    '''
    :param conf_matrix: confusion matrix containing correct and
        predicted labels
    :return: list containing recall values per language
    '''
    recall = {lang : conf_matrix[lang][lang] for lang in range(len(conf_matrix))}
    for idx, rec in enumerate(recall)):    
        recall[rec] = recall[rec] / sum(conf_matrix[idx])

    return recall

def get_precision(conf_matrix):
    '''
    :param conf_matrix: confusion matrix containing correct and
        predicted labels
    :return: list containing precision values per language
    '''
    precision = {lang : conf_matrix[lang][lang] for lang in range(len(conf_matrix))}
    for idx, prec in enumerate(precision)):
        precision[prec] = precision[prec] / sum(conf_matrix.transpose()[idx])

    return precision


def get_fscore(precision, recall):
    '''
    :param precision: precision per language
    :param recall: recall per language
    :return: dictionary representing f-scores per language
    '''
    fscore = defaultdict(float)
    for idx,pre,rec in enumerate(zip(precision, recall)):
        fscore[idx] = 2 * (prec * rec) / (prec + rec)
    return fscore

def get_error_rates(FA, FR, total):
    '''
    :param FA: defaultdict of false acceptances
    :param FR: defaultdict of false rejections
    :return: tuple of defaultdicts representing the False Acceptance Rate FAR,
        False Rejection Rate FRR, and Equal Error Rate EER
    '''
    FAR = FA
    FRR = FR
    EER = defaultdict(float)
    for k1,k2 in zip(FAR, FRR):
        FAR[k1] /= total
        FRR[k2] /= total
        EER[k1] = (FAR[k1] + FRR[k2]) / 2

    return FAR, FRR, EER
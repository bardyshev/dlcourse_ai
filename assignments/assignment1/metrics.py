def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for _ in range(prediction.shape[0]):
        if prediction[_] == 1 and ground_truth[_] == 1:
            TP += 1

        if prediction[_] == 0 and ground_truth[_] == 1:
            FN += 1

        if prediction[_] == 1 and ground_truth[_] == 0:
            FP += 1

        if prediction[_] == 0 and ground_truth[_] == 0:
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return (prediction == ground_truth).sum() / ground_truth.shape[0]

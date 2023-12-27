import numpy as np
from backend.CustomException import CustomException
from sklearn import metrics


MSG_PATH = "MSG_CONSTANT.json"


class ModelValidator:
    y_real: np.array
    y_pred: np.array
    classes: set

    def __init__(self, classes: set = {}, y_real: np.array = np.array([]), y_pred: np.array = np.array([])) -> None:
        self.set_data(classes, y_real, y_pred)

    def set_data(self, classes: set, y_real: np.array, y_pred: np.array) -> None:
        if len(y_real) != len(y_real):
            raise (CustomException(MSG_PATH).update_exception(ValueError(), "YREAL_YPRED_SHAPE_ERROR"))
        if set(y_real.tolist() + y_pred.tolist()).difference(classes):
            raise (CustomException(MSG_PATH).update_exception(ValueError(), "YREAL_YPRED_CLASSES_ERROR"))
        self.y_real = y_real
        self.y_pred = y_pred
        self.classes = classes

    def get_acc(self) -> float:
        acc = sum([1 for y_1, y_2 in zip(self.y_pred, self.y_real) if y_1 == y_2]) / len(self.y_pred)
        return acc

    def get_err(self):
        err = sum([1 for y_1, y_2 in zip(self.y_pred, self.y_real) if y_1 != y_2]) / len(self.y_pred)
        return err

    def get_pr_auc(self):
        raise NotImplementedError("когда-нибудь потом :)")
        #return metrics.precision_recall_fscore_support(self.y_real, self.y_pred, zero_division=1)

    def get_pr(self):

        """
        Compute the Receiver Operating Characteristic Curve for each class (ROC) \
        from predictions.

        :return: dict{[[fpr, tpr],...,]},
        where:
            fpr - false positive rate
            tpr - true positive rate
        """

        unique_class = self.classes
        roc_auc_dict = {}
        for per_class in unique_class:
            if (per_class in self.y_real) or (per_class in self.y_pred):
                other_class = [x for x in unique_class if x != per_class]

                new_actual_class = [0 if x in other_class else 1 for x in self.y_real]
                new_pred_class = [0 if x in other_class else 1 for x in self.y_pred]

                precision, recall, thresholds = metrics.precision_recall_curve(new_actual_class, new_pred_class)
                roc_auc_dict[per_class] = [precision, recall]
            else:
                roc_auc_dict[per_class] = None

        return roc_auc_dict

    def get_roc_auc(self, average="macro"):

        """
        Compute Areas Under the Receiver Operating Characteristic Curve for each class (ROC) \
        from predictions.

        :param average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
            If ``None``, the scores for each class are returned.
            Otherwise, this determines the type of averaging performed on the data.
            Note: multiclass ROC AUC currently only handles the 'macro' and
            'weighted' averages. For multiclass targets, `average=None` is only
            implemented for `multi_class='ovr'` and `average='micro'` is only
            implemented for `multi_class='ovr'`.
            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            ``'samples'``:
                Calculate metrics for each instance, and find their average.
            Will be ignored when ``y_true`` is binary.

        :return: list[square of ROC,...]
        """

        unique_class = self.classes
        roc_auc_dict = {}
        for per_class in unique_class:
            if (per_class in self.y_real) or (per_class in self.y_pred):
                other_class = [x for x in unique_class if x != per_class]

                new_actual_class = [0 if x in other_class else 1 for x in self.y_real]
                new_pred_class = [0 if x in other_class else 1 for x in self.y_pred]

                roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average=average)
                roc_auc_dict[per_class] = roc_auc
            else:
                roc_auc_dict[per_class] = None

        return roc_auc_dict

    def get_roc(self):

        """
        Compute the Receiver Operating Characteristic Curve for each class (ROC) \
        from predictions.

        :return: dict{[[fpr, tpr],...,]},
        where:
            fpr - false positive rate
            tpr - true positive rate
        """

        unique_class = self.classes
        roc_auc_dict = {}
        for per_class in unique_class:
            if (per_class in self.y_real) or (per_class in self.y_pred):
                other_class = [x for x in unique_class if x != per_class]

                new_actual_class = [0 if x in other_class else 1 for x in self.y_real]
                new_pred_class = [0 if x in other_class else 1 for x in self.y_pred]

                fpr, tpr, thresholds = metrics.roc_curve(new_actual_class, new_pred_class)
                roc_auc_dict[per_class] = [fpr, tpr]
            else:
                roc_auc_dict[per_class] = None

        return roc_auc_dict


if __name__ == "__main__":
    m = ModelValidator()

    x = np.array([1, 2, 1, 2, 3])
    y = np.array([1, 2, 1, 3, 3])
    m.set_data({0, 1, 2, 3, 4, 5}, x, y)

    print(m.get_err())







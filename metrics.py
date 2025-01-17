import torch
from torcheval.metrics import R2Score, MulticlassConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score
from monai.metrics import MAEMetric, MSEMetric, ROCAUCMetric

# Custom functions
from utils import list_to_torch_array


class Metrics:
    """
    Class containing functions for computing metric values:
        - Regression: R-squared, MAE, MSE
        - Classification: AUC, F1-score, CE
    """
    def __init__(self, num_ohe_classes, auc_average, f1_average, precision_average, recall_average,
                 f1_zero_division, precision_zero_division, recall_zero_division):
        self.num_ohe_classes = num_ohe_classes
        self.auc_average = auc_average
        self.f1_average = f1_average
        self.precision_average = precision_average
        self.recall_average = recall_average
        self.f1_zero_division = f1_zero_division
        self.precision_zero_division = precision_zero_division
        self.recall_zero_division = recall_zero_division

        self.r2_metric = R2Score()
        self.mae_metric = MAEMetric()
        self.mse_metric = MSEMetric()
        self.auc_metric = ROCAUCMetric(average=self.auc_average)
        self.f1_metric = f1_score
        self.precision_metric = precision_score
        self.recall_metric = recall_score
        self.ce_metric = torch.nn.functional.cross_entropy
        self.conf_matrix = MulticlassConfusionMatrix(self.num_ohe_classes)

    def compute_metric(self, y_pred_list, y_true_list, name):
        # Convert to CPU
        if type(y_pred_list) is list:
            y_pred_list = [x.cpu() for x in y_pred_list]
        if type(y_true_list) is list:
            y_true_list = [x.cpu() for x in y_true_list]

        if name == 'r2':
            return self.compute_r2(y_pred_list=y_pred_list, y_true_list=y_true_list)
        if name == 'mae':
            return self.compute_mae(y_pred_list=y_pred_list, y_true_list=y_true_list)
        elif name == 'mse':
            return self.compute_mse(y_pred_list=y_pred_list, y_true_list=y_true_list)
        elif name == 'auc':
            return self.compute_auc(y_pred_list=y_pred_list, y_true_list=y_true_list)
        elif name == 'f1':
            return self.compute_f1(y_pred_list=y_pred_list, y_true_list=y_true_list, average=self.f1_average)
        elif name == 'precision':
            return self.compute_precision(y_pred_list=y_pred_list, y_true_list=y_true_list,
                                          average=self.precision_average)
        elif name == 'recall':
            return self.compute_recall(y_pred_list=y_pred_list, y_true_list=y_true_list, average=self.recall_average)
        elif name == 'ce':
            return self.compute_ce(y_pred_list=y_pred_list, y_true_list=y_true_list)
        elif name == 'confusion_matrix':
            return self.compute_confusion_matrix(y_pred_list=y_pred_list, y_true_list=y_true_list)
        else:
            raise ValueError('Metric name = {} is not available'.format(name))

    def compute_r2(self, y_pred_list, y_true_list):
        """
        Compute R-squared.
        """
        self.r2_metric.update(input=torch.as_tensor(y_pred_list), target=torch.as_tensor(y_true_list))
        r2_value = self.r2_metric.compute()
        self.r2_metric.reset()
        return r2_value.item()

    def compute_mae(self, y_pred_list, y_true_list):
        """
        Compute Mean Absolute Error.
        """
        # map shape [batch_size, 1] to [batch_size, 1]
        y_pred_list = [x.view(1) if len(x.shape) == 0 else x for x in y_pred_list]
        y_true_list = [x.view(1) if len(x.shape) == 0 else x for x in y_true_list]

        self.mae_metric(y_pred=y_pred_list, y=y_true_list)
        mae_value = self.mae_metric.aggregate()
        self.mae_metric.reset()
        return mae_value.item()

    def compute_mse(self, y_pred_list, y_true_list):
        """
        Compute Mean Squared Error.
        """
        # map shape [batch_size, 1] to [batch_size, 1]
        y_pred_list = [x.view(1) if len(x.shape) == 0 else x for x in y_pred_list]
        y_true_list = [x.view(1) if len(x.shape) == 0 else x for x in y_true_list]

        self.mse_metric(y_pred=y_pred_list, y=y_true_list)
        mse_value = self.mse_metric.aggregate()
        self.mse_metric.reset()
        return mse_value.item()

    def compute_auc(self, y_pred_list, y_true_list):
        """
        Compute Area Under the Curve (AUC).
        """
        self.auc_metric(y_pred=y_pred_list, y=y_true_list)
        auc_value = self.auc_metric.aggregate()
        self.auc_metric.reset()
        return auc_value.item()

    def compute_f1(self, y_pred_list, y_true_list, average):
        """
        Compute the F1 score, also known as balanced F-score or F-measure.
        """
        # Map multidimensional tensor (e.g. shape [batch_size, 3]) to shape [batch_size]
        if (type(y_true_list) == torch.Tensor) and (len(y_true_list.shape) == 2) and (
                y_true_list.shape[-1] == self.num_ohe_classes):
            y_true_list = y_true_list.argmax(dim=-1)
        if (type(y_pred_list) == torch.Tensor) and (len(y_pred_list.shape) == 2) and (
                y_pred_list.shape[-1] == self.num_ohe_classes):
            y_pred_list = y_pred_list.argmax(dim=-1)

        # Map list of multidimensional tensor (e.g. shapes [3]) to shape [] (i.e., tensor constant)
        if ((type(y_true_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_true_list]) == len(y_true_list))):
            y_true_list = [x.argmax(dim=-1) for x in y_true_list]
        if ((type(y_pred_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_pred_list]) == len(y_pred_list))):
            y_pred_list = [x.argmax(dim=-1) for x in y_pred_list]

        return self.f1_metric(y_true=[x.cpu().detach().numpy() for x in y_true_list],
                              y_pred=[x.cpu().detach().numpy() for x in y_pred_list],
                              average=average, zero_division=self.f1_zero_division)

    def compute_precision(self, y_pred_list, y_true_list, average):
        """
        Compute the precision score.
        """
        # Map multidimensional tensor (e.g. shape [batch_size, 3]) to shape [batch_size]
        if (type(y_true_list) == torch.Tensor) and (len(y_true_list.shape) == 2) and (
                y_true_list.shape[-1] == self.num_ohe_classes):
            y_true_list = y_true_list.argmax(dim=-1)
        if (type(y_pred_list) == torch.Tensor) and (len(y_pred_list.shape) == 2) and (
                y_pred_list.shape[-1] == self.num_ohe_classes):
            y_pred_list = y_pred_list.argmax(dim=-1)

        # Map list of multidimensional tensor (e.g. shapes [3]) to shape [] (i.e., tensor constant)
        if ((type(y_true_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_true_list]) == len(y_true_list))):
            y_true_list = [x.argmax(dim=-1) for x in y_true_list]
        if ((type(y_pred_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_pred_list]) == len(y_pred_list))):
            y_pred_list = [x.argmax(dim=-1) for x in y_pred_list]

        return self.precision_metric(y_true=[x.cpu().detach().numpy() for x in y_true_list],
                                     y_pred=[x.cpu().detach().numpy() for x in y_pred_list],
                                     average=average, zero_division=self.precision_zero_division)

    def compute_recall(self, y_pred_list, y_true_list, average):
        """
        Compute the recall score.
        """
        # Map multidimensional tensor (e.g. shape [batch_size, 3]) to shape [batch_size]
        if (type(y_true_list) == torch.Tensor) and (len(y_true_list.shape) == 2) and (
                y_true_list.shape[-1] == self.num_ohe_classes):
            y_true_list = y_true_list.argmax(dim=-1)
        if (type(y_pred_list) == torch.Tensor) and (len(y_pred_list.shape) == 2) and (
                y_pred_list.shape[-1] == self.num_ohe_classes):
            y_pred_list = y_pred_list.argmax(dim=-1)

        # Map list of multidimensional tensor (e.g. shapes [3]) to shape [] (i.e., tensor constant)
        if ((type(y_true_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_true_list]) == len(y_true_list))):
            y_true_list = [x.argmax(dim=-1) for x in y_true_list]
        if ((type(y_pred_list) == list) and
                (sum([len(x) == self.num_ohe_classes for x in y_pred_list]) == len(y_pred_list))):
            y_pred_list = [x.argmax(dim=-1) for x in y_pred_list]

        return self.recall_metric(y_true=[x.cpu().detach().numpy() for x in y_true_list],
                                  y_pred=[x.cpu().detach().numpy() for x in y_pred_list],
                                  average=average, zero_division=self.recall_zero_division)

    def compute_ce(self, y_pred_list, y_true_list):
        """
        Compute cross-entropy (CE).
        """
        # Map list to Torch tensor
        if type(y_pred_list) == list:
            y_pred_list = list_to_torch_array(y_pred_list)
        if type(y_true_list) == list:
            y_true_list = list_to_torch_array(y_true_list)

        return self.ce_metric(y_pred_list.detach().cpu(), y_true_list.detach().cpu())

    def compute_confusion_matrix(self, y_pred_list, y_true_list):
        """
        Compute multi-class confusion matrix.
        """
        if (len(y_true_list.shape) == 2) and (y_true_list.shape[-1] == self.num_ohe_classes):
            y_true_list = y_true_list.argmax(dim=-1)

        self.conf_matrix.update(input=y_pred_list, target=y_true_list)
        cm = self.conf_matrix.compute()
        self.conf_matrix.reset()
        return cm.numpy()


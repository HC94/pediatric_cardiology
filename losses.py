import torch


class FocalLoss(torch.nn.Module):
    """
    The Focal loss addresses class imbalance during training.
    """
    def __init__(self, num_ohe_classes, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.num_ohe_classes = num_ohe_classes
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = self.one_hot(index=target, classes=self.num_ohe_classes)  # input.size(-1))
        logit = torch.nn.functional.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0)
        index = index.view(*view)
        ones = 1.

        if isinstance(index, torch.autograd.Variable):
            ones = torch.as_tensor(torch.Tensor(index.size()).fill_(1), device=index.device)
            mask = torch.as_tensor(mask, device=index.device)

        return mask.scatter_(1, index, ones)


class F1Loss(torch.nn.Module):
    """
    F1 loss.
    """
    def __init__(self, num_ohe_classes, epsilon=1e-7):
        super().__init__()
        self.num_ohe_classes = num_ohe_classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = torch.nn.functional.one_hot(y_true, self.num_ohe_classes).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class L1Loss(torch.nn.Module):
    """
    L1 loss
    """
    def __init__(self, num_ohe_classes, reduction):
        super().__init__()
        self.num_ohe_classes = num_ohe_classes
        self.l1 = torch.nn.SmoothL1Loss(reduction=reduction, beta=0.05)
        self.softmax_act = torch.nn.Softmax(dim=-1)

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(
            self.l1(self.softmax_act(y_pred),
                    torch.nn.functional.one_hot(y_true, num_classes=self.num_ohe_classes).float()))
        return loss


class RankingLoss(torch.nn.Module):
    """
    In self.rank(): if y = 1 then it assumed the first input should be ranked higher (have a larger value) than the
    second input, and vice-versa for y = -1.
    """
    def __init__(self, reduction):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.rank = torch.nn.MarginRankingLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)
        x1 = y_pred[:, 0]
        x2 = y_pred[:, 1]
        y = - (y_true - 0.5) * 2
        loss = self.rank(x1, x2, y)
        return loss


class SoftAUCLoss(torch.nn.Module):
    """
    Soft version of AUC that uses Wilcoxon-Mann-Whitney U statistic.
    """
    def __init__(self):
        super().__init__()
        self.softmax_act = torch.nn.Softmax(dim=-1)
        self.sigmoid_act = torch.nn.Sigmoid()
        self.gamma = 0.2  # 0.7
        self.p = 3

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = self.softmax_act(y_pred)[:, 1]
        # Get the predictions of all the positive and negative examples
        pos = y_pred[y_true.bool()].view(1, -1)
        neg = y_pred[~y_true.bool()].view(-1, 1)
        # Compute Wilcoxon-Mann-Whitney U statistic
        difference = torch.zeros_like(pos * neg) + pos - neg - self.gamma
        masked = difference[difference < 0.0]
        loss = torch.sum(torch.pow(-masked, self.p))
        return loss


class CustomLoss(torch.nn.Module):
    """
    Combine loss functions.
    """
    def __init__(self, loss_function_weights, label_weights, num_ohe_classes,
                 reduction, label_smoothing, focal_loss_gamma):
        super().__init__()
        self.loss_function_weights = loss_function_weights

        self.ce = torch.nn.CrossEntropyLoss(weight=label_weights, reduction=reduction, label_smoothing=label_smoothing)
        self.f1 = F1Loss(num_ohe_classes=num_ohe_classes)
        self.focal = FocalLoss(num_ohe_classes=num_ohe_classes, gamma=focal_loss_gamma)
        self.l1 = L1Loss(num_ohe_classes=num_ohe_classes, reduction=reduction)
        self.ranking = RankingLoss(reduction=reduction)
        self.soft_auc = SoftAUCLoss()

    def forward(self, y_pred, y_true):
        fcns = [self.ce, self.f1, self.focal, self.l1, self.ranking, self.soft_auc]

        loss = 0
        for i, (w, fcn) in enumerate(zip(self.loss_function_weights, fcns)):
            if w != 0:
                loss = loss + w * fcn(y_pred, y_true)

        return loss


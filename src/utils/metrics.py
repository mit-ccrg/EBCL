import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.loss import RMSELoss

metric_names = [
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "fdr",
    "ppv",
    "f1",
    "auc",
    "apr",
    "acc",
    "loss",
]
score_to_dict = lambda name, score: dict((name[i], score[i]) for i in range(len(score)))


class Evaluator(object):
    """Evaluator Object for
    prediction performance"""

    def __init__(self, args=None):
        if args != None:
            self.args = args
            self.batch_size = args.batch_size
        self.confusion_matrix = np.zeros((2, 2))
        self.logits = []
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.loss = 0
        self.threshold = 0.5
        # self.rmse = RMSELoss(args)

        # for statistical testing
        self.loss_list = []
        self.auc_list = []
        self.true_classes = []
        self.pre_class_preds = []
        self.post_class_preds = []
        self.pre_probs = []
        self.post_probs = []
        self.avg_logit = []
        self.pre_embed_avg = []
        self.post_embed_avg = []

    def add_batch(self, logits, post_embed_avg, pre_embed_avg):
        """Stores classification metric data,

        Given batch clip logits, computes the classification probababilities
        and class predictions for abatch
        Args:
            logits: matrix of post_embeddings @ pre_embeddings
                (axis_0 = subjective data index, axis_1 = objective data index
        """
        # compute class probabilities
        pre_probs = torch.nn.functional.softmax(logits, dim=1).numpy()
        post_probs = torch.nn.functional.softmax(logits.T, dim=1).numpy()

        # class predictions, for accuracy and auc computation
        pre_class_preds = np.argmax(pre_probs, axis=1)
        post_class_preds = np.argmax(post_probs, axis=1)

        # class ground truth, for accuracy computation
        true_class = np.arange(logits.shape[0])

        # for auc get sigmoid
        self.true_classes.append(true_class)

        self.pre_class_preds.append(pre_class_preds)
        self.post_class_preds.append(post_class_preds)

        self.pre_probs.append(pre_probs)
        self.post_probs.append(post_probs)

        self.avg_logit.append(np.mean(logits.numpy()))

        self.post_embed_avg.append(post_embed_avg.numpy())
        self.pre_embed_avg.append(pre_embed_avg.numpy())

    def performance_metric(self, validation=True):
        """Collect AUC, Accuracy, and Loss
        AUC is calculated using One vs Rest macro-average:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

        Returns:
            pre_auc: auc for objective clip classification
            pre_accuracy: accuracy for objective clip classification
            post_auc: auc for subjective clip classification
            post_accuracy: accuracy for subjective clip classification
            np.mean(self.avg_logit): average normalized logit value
                (should be in [-e**model.t.weight, e**model.t.weight])
            post_embed_avg: average unnormalized subjective embedding value
            pre_embed_avg: average unnormalized objective embedding value

        """
        assert (
            validation
        ), "Only validation performance is supported, this will take too long on training data"
        true_classes = np.concatenate(self.true_classes)
        # print(f"true_classes.shape: {true_classes.shape}")
        pre_class_preds = np.concatenate(self.pre_class_preds)
        post_class_preds = np.concatenate(self.post_class_preds)
        pre_probs = np.vstack(self.pre_probs)
        post_probs = np.vstack(self.post_probs)

        pre_auc = roc_auc_score(
            true_classes, pre_probs, multi_class="ovr", average="macro"
        )
        pre_accuracy = accuracy_score(true_classes, pre_class_preds)

        post_auc = roc_auc_score(
            true_classes, post_probs, multi_class="ovr", average="macro"
        )
        post_accuracy = accuracy_score(true_classes, post_class_preds)

        post_embed_avg = np.mean(self.post_embed_avg)
        pre_embed_avg = np.mean(self.pre_embed_avg)

        return (
            pre_auc,
            pre_accuracy,
            post_auc,
            post_accuracy,
            np.mean(self.avg_logit),
            post_embed_avg,
            pre_embed_avg,
        )

    def add_task_batch(self, y_true, y_pred, loss):
        self.y_true.append(y_true.squeeze())
        self.y_pred_proba.append(y_pred.squeeze())

        if self.args.train_mode == "regression":
            self.y_pred.append(y_pred.squeeze())
            self.loss = loss

        elif self.args.train_mode == "binary_class":
            self.y_pred.append(np.array(y_pred > self.threshold).astype(int).squeeze())
            self.confusion_matrix += confusion_matrix((y_pred > self.threshold), y_true)

    def performance_task(self, validation=False):
        y_true = np.hstack(self.y_true)
        y_pred_proba = np.hstack(self.y_pred_proba)
        y_pred = np.hstack(self.y_pred)

        if self.args.train_mode == "regression":
            loss = self.loss
            r, pval = stats.pearsonr(
                np.concatenate(self.y_true).squeeze(),
                np.concatenate(self.y_pred).squeeze(),
            )

            if self.args.bootstrap and not validation:
                loss_list, r_list, pval_list = self.do_bootstrap_regression(
                    np.concatenate(self.y_pred), np.concatenate(self.y_true)
                )
                loss_lower, loss_upper = self.confidence_interval(loss_list)
                r_lower, r_upper = self.confidence_interval(r_list)
                pval_lower, pval_upper = self.confidence_interval(pval_list)
                self.loss_list = loss_list

                return (
                    np.mean(loss_list),
                    np.mean(r_list),
                    np.mean(pval_list),
                    (loss_lower, loss_upper),
                    (r_lower, r_upper),
                    (pval_lower, pval_upper),
                )

            return loss, r, pval

        elif self.args.train_mode == "binary_class":
            auc = roc_auc_score(y_true, y_pred_proba)
            apr = average_precision_score(y_true, y_pred_proba)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            if self.args.bootstrap and not validation:
                auc_list, apr_list, acc_list, f1_list = self.do_bootstrap(
                    y_pred_proba, y_pred, y_true
                )
                f1_lower, f1_upper = self.confidence_interval(f1_list)
                auc_lower, auc_upper = self.confidence_interval(auc_list)
                apr_lower, apr_upper = self.confidence_interval(apr_list)
                acc_lower, acc_upper = self.confidence_interval(acc_list)
                self.auc_list = auc_list
                self.apr_list = apr_list

                return (
                    np.mean(auc_list),
                    (auc_lower, auc_upper),
                    np.mean(apr_list),
                    (apr_lower, apr_upper),
                    np.mean(acc_list),
                    (acc_lower, acc_upper),
                )

            return f1, auc, apr, acc

    def get_datamap_score(self, y_pred_prob_epochs, y_true, threshold=0.2):
        """
        Unimodal baseline
        Parameters:
            y_pred_prob_epochs: Vector of size N X C X Epochs,
                where N=size of train set, C=Number of classes,
                Epochs=number of training epochs
            y_true: True class, 0-indexed


        Returns:
            datamap scores, np.array of floats.
        """
        try:
            assert np.min(self.y_true) == 0
        except:
            raise NotImplementedError

        # getting scores over epochs
        instance_arr = []

        for i in range(y_true):
            true_class_probs = y_pred_prob_epochs[i, y_true[i], :]
            instance_arr.append(true_class_probs)

        mean_scores = np.mean(instance_arr, axis=1)
        var_scores = np.std(instance_arr, axis=1)

        datamap_scores = []
        for i in range(y_true):
            curr_score = 0
            if mean_scores[i] < threshold and var_scores[i] < threshold:
                curr_score = 1

            datamap_scores.append(curr_score)
        return datamap_scores

    def get_aum_score(y_pred_prob_epochs, y_true, threshold=0.2, thresholding=False):
        """
        Parameters:
            y_pred_prob_epochs: Vector of size N X C X Epochs,
                where N=size of train set, C=Number of classes,
                Epochs=number of training epochs
            y_true: True class, 0-indexed


        Returns:
            AUM scores, np.array of floats.
        """
        try:
            assert np.min(y_true) == 0
        except:
            raise NotImplementedError

        # getting scores over epochs
        aum_scores = []
        for i in range(y_true):
            true_class_probs = y_pred_prob_epochs[i, y_true[i], :]
            curr_margins = []
            for epoch in range(y_pred_prob_epochs.shape[2]):
                curr_margins.append(
                    true_class_probs[epoch] - np.max(y_pred_prob_epochs[i, :, epoch])
                )

            if thresholding:
                aum_scores.append(np.mean(curr_margins) > threshold)
            else:
                aum_scores.append(np.mean(curr_margins))

        return aum_scores

    def metric_performance(self):
        features = np.concatenate(self.features, axis=0)
        try:
            target_labels = np.array(self.y_true)
        except:
            target_labels = torch.cat(self.y_true, dim=0).numpy()
        return self.recall_at_1(target_labels, features), self.nmi(
            target_labels, features
        )

    def do_bootstrap(self, preds, pred_vals, trues, n=1000):
        auc_list = []
        apr_list = []
        acc_list = []
        f1_list = []

        rng = np.random.RandomState(seed=1)
        for _ in range(n):
            idxs = rng.choice(len(trues), size=len(trues), replace=True)
            pred_arr = preds[idxs]
            true_arr = trues[idxs]
            pred_val_arr = pred_vals[idxs]

            auc = roc_auc_score(true_arr, pred_arr)
            apr = average_precision_score(true_arr, pred_arr)
            acc = accuracy_score(true_arr, pred_val_arr)
            f1 = f1_score(true_arr, pred_val_arr)

            auc_list.append(auc)
            apr_list.append(apr)
            acc_list.append(acc)
            f1_list.append(f1)

        return (
            np.array(auc_list),
            np.array(apr_list),
            np.array(acc_list),
            np.array(f1_list),
        )

    def do_bootstrap_regression(self, preds, trues, n=1000):
        rmse_list = []
        r_list = []
        pval_list = []

        rng = np.random.RandomState(seed=1)
        for _ in range(n):
            idxs = rng.choice(len(trues), size=len(trues), replace=True)
            pred_arr = preds[idxs]
            true_arr = trues[idxs]

            rmse = self.rmse(torch.tensor(pred_arr), torch.tensor(true_arr))
            r, pval = stats.pearsonr(true_arr, pred_arr)

            rmse_list.append(rmse)
            r_list.append(r)
            pval_list.append(pval)

        return np.array(rmse_list), np.array(r_list), np.array(pval_list)

    def confidence_interval(self, values, alpha=0.95):
        lower = np.percentile(values, (1 - alpha) / 2 * 100)
        upper = np.percentile(values, (alpha + (1 - alpha) / 2) * 100)
        return lower, upper

    def return_pred(self):
        return self.y_pred, self.y_true

    def reset(self):
        self.confusion_matrix = np.zeros((2,) * 2)
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.features = []

        self.loss_list = []
        self.auc_list = []
        self.true_classes = []
        self.pre_class_preds = []
        self.post_class_preds = []
        self.pre_probs = []
        self.post_probs = []
        self.avg_logit = []
        self.pre_embed_avg = []
        self.post_embed_avg = []

        self.loss = 0.0

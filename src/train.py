import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, EvalPrediction, AutoTokenizer, set_seed, Trainer
import yaml
from os import path, makedirs, listdir
import json

#### UTILS ####
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score, precision_score, recall_score, jaccard_score, matthews_corrcoef, multilabel_confusion_matrix, zero_one_loss
from torch import sigmoid, Tensor, stack, from_numpy
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.ops import focal_loss
from transformers import Trainer
from torch import nn, Tensor, nonzero, sort
import pickle
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
    
class CustomTrainer(Trainer):
    """
    Custom Trainer to compute the weighted BCE loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights = None
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2

    def prepare_labels(self, data_path, language, seed, device):
        """
        Set the mlb encoder and the weights for the BCE loss.

        :param data_path: Path to the data.
        :param language: Language of the data.
        :param seed: Seed of the data.
        :param device: Device to use.
        """
        # Load the weights
        with open(os.path.join(data_path, language, f"split_{seed}", "train_labs_count.json"), "r") as weights_fp:
            data = json.load(weights_fp)
            weights = []
            """ # Approach with max weight in case of 0
            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key])/data["labels"][key] if data["labels"][key] != 0 else None)
            
            # If the weight is None, set it to the maximum weight
            max_weight = max([w for w in weights if w is not None])
            weights = [w if w else max_weight for w in weights] """

            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key] + 1e-10)/(data["labels"][key] + 1e-10))

            self.custom_weights = Tensor(weights).to(device)
        
    def set_weighted_loss(self):
        """
        Set the loss to the weighted BCE loss.
        """
        self.use_focal_loss = False

    def set_focal_params(self, alpha, gamma):
        """
        Set the focal loss parameters.

        :param alpha: Alpha parameter.
        :param gamma: Gamma parameter.
        """
        self.focal_alpha = alpha
        self.focal_gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function to compute either the focal loss or the weighted BCE loss.

        :param model: Model to use.
        :param inputs: Inputs to the model.
        :param return_outputs: Whether to return the outputs. Default: False.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits = outputs.get("logits")
            
            if self.use_focal_loss:
                loss = focal_loss.sigmoid_focal_loss(
                    logits,
                    labels,
                    reduction="mean",
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                    )
            else:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.custom_weights)
                loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

def sklearn_metrics_core(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False):
    """
    Shared code for the sklearn metrics.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :param threshold: Threshold to use for the predictions.
    :param get_conf_matrix: Whether to get the confusion matrix.
    :param get_class_report: Whether to get the classification report.
    :return: Initialized variables.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    if get_conf_matrix:
        with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
            mlb_encoder = pickle.load(f)
        
        # labels = mlb_encoder.inverse_transform(np.ones((1, y_true.shape[1])))[0]
        labels = mlb_encoder.classes_.tolist()
        mlb_conf = multilabel_confusion_matrix(y_true, y_pred)
        conf_matrix = {}
        for i in range(len(labels)):
            conf_matrix[labels[i]] = mlb_conf[i].tolist()
    else:
        conf_matrix = None
    
    if get_class_report:
        class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, digits=4)
        class_report = {
            key: value for key, value in class_report.items() if key.isnumeric()# and value['support'] > 0
        }
    else:
        class_report = None

    to_return = {}

    return probs, y_pred, conf_matrix, class_report, to_return


def sklearn_metrics_full(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False, parent_handling="none"):
    """
    Return all the metrics and the classification report for the predictions.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions.
    :param get_conf_matrix: If True, return the confusion matrix.
    :param get_class_report: If True, return the classification report.
    :param parent_handling: How to handle the parent labels.
    :return: A dictionary with the metrics and a classification report.
    """
    probs, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    to_return = calculate_metrics(y_true, y_pred, probs, to_return)

    if parent_handling == "add" or parent_handling == "builtin":
        to_return.update(calculate_parent_metrics(y_true, predictions, data_path, parent_handling))

    return to_return, class_report, conf_matrix


def sklearn_metrics_single(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False,
    eval_metric=''):
    """
    Return the specified metric and the classification report for the predictions during the training.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions.
    :param get_conf_matrix: If True, return the confusion matrix.
    :param get_class_report: If True, return the classification report.
    :param eval_metric: The metric to use for the evaluation.
    :return: A dictionary with the metric and a classification report.
    """
    _, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    if "accuracy" in eval_metric:
        to_return["accuracy"] = accuracy_score(y_true, y_pred)
    elif "f1" in eval_metric:
        to_return[eval_metric] = f1_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "precision" in eval_metric:
        to_return[eval_metric] = precision_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "recall" in eval_metric:
        to_return[eval_metric] = recall_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "hamming" in eval_metric:
        to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    elif "jaccard" in eval_metric:
        to_return[eval_metric] = jaccard_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "matthews" in eval_metric:
        references = np.array(y_true)
        predictions = np.array(y_pred)
        if eval_metric == "matthews_micro":
            to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
        elif eval_metric == "matthews_macro":
            to_return["matthews_macro"] = np.mean([
                matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
                for i in range(references.shape[1])
            ])
    elif "roc_auc" in eval_metric:
        to_return[eval_metric] = roc_auc_score(y_true, y_pred, average=eval_metric.split("_")[1])
    elif "ndcg" in eval_metric:
        to_return[eval_metric] = ndcg_score(y_true, y_pred, k=eval_metric.split("_")[1])

    return to_return, class_report, conf_matrix


def calculate_metrics(y_true, y_pred, probs, to_return):
    """
    Calculates the metrics.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param probs: Predicted probabilities.
    :param to_return: Dictionary to return.
    """
    averaging=["micro", "macro", "weighted", "samples"]

    to_return["accuracy"] = accuracy_score(y_true, y_pred)
    
    true_labels = [nonzero(Tensor(labels), as_tuple=True)[0] for labels in y_true]
    pred_labels = sort(probs, descending=True)[1][:, :6]
    pk_scores = [np.intersect1d(true, pred).shape[0] / (pred.shape[0] + 1e-10) for true, pred in
                    zip(true_labels, pred_labels)]
    rk_scores = [np.intersect1d(true, pred).shape[0] / (true.shape[0] + 1e-10) for true, pred in
                    zip(true_labels, pred_labels)]
    f1k_scores = [2 * recall * precision / (recall + precision + 1e-10) for recall, precision in zip(pk_scores, rk_scores)]
    to_return["f1"] = sum(f1k_scores) / len(f1k_scores)
    for avg in averaging:
        to_return[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    
    to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    
    for avg in averaging:
        to_return[f"jaccard_{avg}"] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
    
    references = np.array(y_true)
    predictions = np.array(y_pred)
    to_return["matthews_macro"] = np.mean([
        matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
        for i in range(references.shape[1])
    ])
    to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
    
    for avg in averaging:
        try:
            to_return[f"roc_auc_{avg}"] = roc_auc_score(y_true, y_pred, average=avg)
        except ValueError:
            to_return[f"roc_auc_{avg}"] = 0.0
    
    for k in [1, 3, 5, 10]:
        to_return[f"ndcg_{k}"] = ndcg_score(y_true, y_pred, k=k)

    to_return["zero_one_loss"] = int(zero_one_loss(y_true, y_pred, normalize=False))

    to_return["total_samples"] = len(y_true)

    return to_return


def calculate_parent_metrics(y_true, predictions, data_path, mode):
    """
    Get the parent label metrics.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :return: Metrics.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= 0.5).astype(int)

    # Get the labels
    if mode == "add":
        mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred = add_setup(data_path, y_true, y_pred)
    else:
        mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred = builtin_setup(data_path, y_true, y_pred)

    # create the lists to use to calculate the F1 score
    true_labels = [nonzero(Tensor(labels), as_tuple=True)[0] for labels in y_true]
    (mt_labels_true_manual, mt_labels_pred_manual,
     do_labels_true_manual, do_labels_pred_manual) = initialize_manual_labels(true_labels, probs)
    
    # convert the dictionaries to lists with only the values
    mt_labels_true = [list(mt_labels_true[i].values()) for i in range(len(mt_labels_true))]
    mt_labels_pred = [list(mt_labels_pred[i].values()) for i in range(len(mt_labels_pred))]
    do_labels_true = [list(do_labels_true[i].values()) for i in range(len(do_labels_true))]
    do_labels_pred = [list(do_labels_pred[i].values()) for i in range(len(do_labels_pred))]
        
    metrics = {}

    for label_type in ["mt", "do"]:
        labels_true = mt_labels_true if label_type == "mt" else do_labels_true
        labels_pred = mt_labels_pred if label_type == "mt" else do_labels_pred

        new_metrics = calculate_metrics(labels_true, labels_pred, probs, {})

        # Calculate the F1 score
        # It needs to be redone here because the labels are in simple arrays while in 'calculate_metrics' it uses tensors
        labels_true = mt_labels_true_manual if label_type == "mt" else do_labels_true_manual
        labels_pred = mt_labels_pred_manual if label_type == "mt" else do_labels_pred_manual
        pk_scores = []
        rk_scores = []
        for true, pred in zip(labels_true, labels_pred):
            pk_scores.append(np.intersect1d(true, pred).shape[0] / (len(pred) + 1e-10))
            rk_scores.append(np.intersect1d(true, pred).shape[0] / (len(true) + 1e-10))
        f1k_scores = [2 * recall * precision / (recall + precision + 1e-10) for recall, precision in zip(pk_scores, rk_scores)]
        new_metrics["f1"] = sum(f1k_scores) / len(f1k_scores)

        keys = [key + f"_{label_type}" for key in list(new_metrics)]

        to_update = {key: new_metrics[key.replace(f"_{label_type}", "")] for key in keys}
        
        metrics.update(to_update)

    return metrics

def initialize_manual_labels(true_labels, probs):
    """
    Initialize the arrays of labels to be to calculate the F1 for the parent labels.

    :param true_labels: Array with the true labels in the format:
        [
            ["3254", "4567", "2728", ...],
            ...
        ]
    :param probs: Predicted probabilities.
    :return: Arrays with the true and predicted MT and DO labels in the format:
        [
            ["3254", "4567", "2728", ...],
            ...
        ]
    """
    with open("./config/mt_labels.json", "r") as fp:
        mt_mapping = json.load(fp)

    pred_labels = sort(probs, descending=True)[1][:, :5]

    true_labels_mt = []
    true_labels_do = []
    for labels in true_labels:
        # Every label should be present in the mapping, only the label "eurovoc",
        # identified by id "3712", is not present and has no MT mapping.
        # See https://op.europa.eu/en/web/eu-vocabularies/concept/-/resource?uri=http://eurovoc.europa.eu/3712
        to_append_mt = []
        to_append_do = []
        for label in labels:
            if str(label.item()) in mt_mapping:
                to_append_mt.append(mt_mapping[str(label.item())])
                to_append_do.append(mt_mapping[str(label.item())][:2])
        true_labels_mt.append(to_append_mt)
        true_labels_do.append(to_append_do)
    
    pred_labels_mt = []
    pred_labels_do = []
    for labels in pred_labels:
        to_append_mt = []
        to_append_do = []
        for label in labels:
            if str(label.item()) in mt_mapping:
                to_append_mt.append(mt_mapping[str(label.item())])
                to_append_do.append(mt_mapping[str(label.item())][:2])
        pred_labels_mt.append(to_append_mt)
        pred_labels_do.append(to_append_do)

    return true_labels_mt, pred_labels_mt, true_labels_do, pred_labels_do

def add_setup(data_path, y_true, y_pred):
    """
    Initialize the parent labels by adding the parents artificially
    (the Thesaurus Concept labels are mapped to their parent Micro Thesaurus and Domain labels).

    :param data_path: Path to the data.
    :param y_true: True labels.
    :param y_pred: Predictions.
    :return: Labels.
    """
    with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
        mlb_encoder = pickle.load(f)
        
        labels_true = mlb_encoder.inverse_transform(y_true)
        labels_pred = mlb_encoder.inverse_transform(y_pred)

        # load the file "mt_position" from the config folder and the mapping
        with open("./config/mt_labels_position.json", "r") as fp:
            mt_position = json.load(fp)
        with open("./config/mt_labels.json", "r") as fp:
            mt_mapping = json.load(fp)
        # initialize a dictionary for both true and pred labels
        mt_labels_true = [{k:0 for k in mt_position} for _ in range(len(labels_true))]
        mt_labels_pred = [{k:0 for k in mt_position} for _ in range(len(labels_pred))]
        # if the label is present, set the value to 1
        for i in range(len(labels_true)):
            for label in labels_true[i]:
                if label in mt_mapping:
                    mt_labels_true[i][mt_mapping[label]] = 1
        for i in range(len(labels_pred)):
            for label in labels_pred[i]:
                if label in mt_mapping:
                    mt_labels_pred[i][mt_mapping[label]] = 1
        # load the file "do_labels_position" from the config folder
        with open("./config/domain_labels_position.json", "r") as fp:
            do_position = json.load(fp)
        # initialize a dictionary for both true and pred labels
        do_labels_true = [{k:0 for k in do_position} for _ in range(len(labels_true))]
        do_labels_pred = [{k:0 for k in do_position} for _ in range(len(labels_pred))]
        # if the label is present, set the value to 1
        for i in range(len(mt_labels_true)):
            for label in mt_labels_true[i]:
                if mt_labels_true[i][label] == 1:
                    do_labels_true[i][label[:2]] = 1
        for i in range(len(mt_labels_pred)):
            for label in mt_labels_pred[i]:
                if mt_labels_pred[i][label] == 1:
                    do_labels_pred[i][label[:2]] = 1
    
    return mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred

def builtin_setup(data_path, y_true, y_pred):
    """
    Initialize the parent labels if the parents are already present in the training data
    (only if the data for the model was processed with the --add_mt_do flag).

    :param data_path: Path to the data.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Labels.
    """
    with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
        mlb_encoder = pickle.load(f)
        
        labels_true = mlb_encoder.inverse_transform(y_true)
        labels_pred = mlb_encoder.inverse_transform(y_pred)

        # load the file "mt_position" from the config folder and the mapping
        with open("./config/mt_labels_position.json", "r") as fp:
            mt_position = json.load(fp)
        # load the file "domain_labels_position" from the config folder
        with open("./config/domain_labels_position.json", "r") as fp:
            do_position = json.load(fp)

        # initialize a dictionary for both true and pred labels
        mt_labels_true = [{k:0 for k in mt_position} for _ in range(len(labels_true))]
        mt_labels_pred = [{k:0 for k in mt_position} for _ in range(len(labels_pred))]
        do_labels_true = [{k:0 for k in do_position} for _ in range(len(labels_true))]
        do_labels_pred = [{k:0 for k in do_position} for _ in range(len(labels_pred))]

        # if the label is present, set the value to 1
        for i in range(len(labels_true)):
            for label in labels_true[i]:
                if "_mt" in label:
                    mt_labels_true[i][label.split("_mt")[0]] = 1
                elif "_do" in label:
                    do_labels_true[i][label.split("_do")[0]] = 1
        for i in range(len(labels_pred)):
            for label in labels_pred[i]:
                if "_mt" in label:
                    mt_labels_pred[i][label.split("_mt")[0]] = 1
                elif "_do" in label:
                    do_labels_pred[i][label.split("_do")[0]] = 1
    
    return mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred

def data_collator_tensordataset(features):
    """
    Custom data collator for datasets of the type TensorDataset.

    :param features: List of features.
    :return: Batch.
    """
    batch = {}
    batch['input_ids'] = stack([f[0] for f in features])
    batch['attention_mask'] = stack([f[1] for f in features])
    batch['labels'] = stack([f[2] for f in features])
    
    return batch


def load_data(data_path, lang, data_type, seed):
    """
    Load the data from the specified directory.

    :param data_path: Path to the data.
    :param lang: Language.
    :param data_type: Type of data to load (train or test).
    :param seed: Seed to load.
    :return: List of train, dev and test loaders.
    """
    to_return = []

    for directory in os.listdir(data_path):
        if lang == directory:
            if data_type == "train":
                print("\nLoading training and dev data from directory {}...".format(os.path.join(data_path, directory, f"split_{seed}")))
    
                # The data is stored in numpy arrays, so it has to be converted to tensors.
                train_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_X.npy")))
                train_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_mask.npy")))
                train_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_y.npy"))).float()
    
                assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]
    
                dev_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_X.npy")))
                dev_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_mask.npy")))
                dev_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_y.npy"))).float()
    
                assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]
    
                dataset_train = TensorDataset(train_X, train_mask, train_y)
    
                dataset_dev = TensorDataset(dev_X, dev_mask, dev_y)
                to_return = [dataset_train, dataset_dev, train_y.shape[1]]
    
            elif data_type == "test":
                print("\nLoading test data from directory {}...".format(os.path.join(data_path, directory, f"split_{seed}")))
                test_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_X.npy")))
                test_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_mask.npy")))
                test_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_y.npy"))).float()
    
                assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]
    
                dataset_test = TensorDataset(test_X, test_mask, test_y)
    
                to_return = dataset_test
            break

    return to_return
#### UTILS ####


language = ""
current_epoch = 0
current_split = 0

src_basepath = "/shared/src"
file_basepath = "/files"

def get_metrics(y_true, predictions, args, language, current_split):
    """
    Return the metrics for the predictions.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: Dictionary with the metrics.
    """

    global current_epoch
    
    metrics, class_report, _ = sklearn_metrics_full(
        y_true,
        predictions,
        "",
        args.threshold,
        False,
        args.save_class_report,
    ) if args.full_metrics else sklearn_metrics_single(
        y_true,
        predictions,
        "",
        args.threshold,
        False,
        args.save_class_report,
        eval_metric=args.eval_metric,
    )

    if args.save_class_report:
        if current_epoch % args.class_report_step == 0:
            with open(path.join(
                args.models_path,
                language,
                str(current_split),
                "train_reports",
                f"class_report_{current_epoch}.json",
            ), "w") as class_report_fp:
                class_report.update(metrics)
                json.dump(class_report, class_report_fp, indent=2)

    current_epoch += 1

    metrics_path = path.join(
                args.models_path,
                language,
                str(current_split),
                "metrics.json",
            )
    print(f"Saving metrics at {metrics_path} ...") 
    with open(metrics_path, "w") as metrics_fp:
                json.dump(metrics, metrics_fp, indent=2)

    
    return metrics


def start_train(args):
    """
    Launch the training of the models.
    """
    # Load the configuration for the models of all languages
    with open(f"{src_basepath}/config/models.yml", "r") as config_fp:
        config = yaml.safe_load(config_fp)

    # Load the seeds for the different splits
    if args.seeds != "all":
        seeds = args.seeds.split(",")
    else:
        seeds = [name.split("_")[1] for name in listdir(path.join(args.data_path, args.lang)) if "split" in name]

    print(f"Working on device: {args.device}")

    print(f"\nArguments: {vars(args)}")

    # Create the directory for the models
    if not path.exists(args.models_path):
        makedirs(args.models_path)

    language = args.lang

    print(f"\nTraining for language: '{args.lang}' using: '{config[args.lang]}'...")

    res = {}
    
    # Train the models for all splits
    for seed in seeds:
        current_split = seed

        # Load the data
        train_set, dev_set, num_classes = load_data(args.data_path, args.lang, "train", seed)

        # Create the directory for the models of the current language
        makedirs(path.join(args.models_path, args.lang,
                    seed), exist_ok=True)

        # Create the directory for the classification report of the current language
        if args.save_class_report:
            makedirs(path.join(args.models_path, args.lang, str(
                seed), "train_reports"), exist_ok=True)

        set_seed(int(seed))

        tokenizer = AutoTokenizer.from_pretrained(config[args.lang])

        with open(path.join(args.data_path, language, f"split_{seed}", "train_labs_count.json"), "r") as weights_fp:
            data = json.load(weights_fp)
            labels = list(data["labels"].keys())

        model = AutoModelForSequenceClassification.from_pretrained(
            config[args.lang],
            problem_type="multi_label_classification",
            num_labels=num_classes,
            id2label={id_label:label for id_label, label in enumerate(labels)},
            label2id={label:id_label for id_label, label in enumerate(labels)},
            trust_remote_code=args.trust_remote,
        )

        # If the device specified via the arguments is "cpu", avoid using CUDA
        # even if it is available
        no_cuda = True if args.device == "cpu" else False

        # Create the training arguments.
        train_args = TrainingArguments(
            path.join(args.models_path, args.lang, seed),
            evaluation_strategy="epoch",
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.epochs,
            lr_scheduler_type="linear",
            warmup_steps=len(train_set),
            logging_strategy="epoch",
            logging_dir=path.join(
                args.models_path, args.lang, seed, 'logs'),
            save_strategy="epoch",
            no_cuda=no_cuda,
            seed=int(seed),
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model=args.eval_metric,
            optim="adamw_torch",
            optim_args="correct_bias=True",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=0.01,
            report_to="all",
            fp16=args.fp16,
        )

        def compute_metrics(p: EvalPrediction):
            """
            Compute the metrics for the predictions during the training.
        
            :param p: EvalPrediction object.
            :return: Dictionary with the metrics.
            """
            preds = p.predictions[0] if isinstance(
                p.predictions, tuple) else p.predictions
            result = get_metrics(p.label_ids, preds, args, language, current_split)
            return result
            
        # Create the trainer. It uses a custom data collator to convert the
        # dataset to a compatible dataset.

        if args.custom_loss:
            trainer = CustomTrainer(
                model,
                train_args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )

            trainer.prepare_labels(
                args.data_path, args.lang, seed, args.device)

            if args.weighted_loss:
                trainer.set_weighted_loss()
        else:
            trainer = Trainer(
                model,
                train_args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )
        trainer.train()
        res[seed] = path.join(args.models_path, args.lang, seed, trainer.state.best_model_checkpoint)
        # print(f"Best checkpoint path: {trainer.state.best_model_checkpoint}")
    return res
    
def train(project, 
          train_data,
          lang = "it", 
          seeds = "110",
          device = "cuda",
          epochs = 100,
          batch_size = 8,
          learning_rate = 3e-5,
          max_grad_norm = 5,
          threshold = 0.5,
          custom_loss = False,
          weighted_loss = False,
          fp16 = False,
          eval_metric = "f1_micro",
          full_metrics = False,
          trust_remote = False,
          save_class_report = False,
          class_report_step = 1
         ):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="it", help="Language to train the model on.")
    parser.add_argument("--seeds", type=str, default="all", help="Seeds to be used to load the data splits, separated by a comma (e.g. 110,221). Use 'all' to use all the data splits.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to train on.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the prediction confidence.")
    parser.add_argument("--custom_loss", action="store_true", default=False, help="Enable the custom loss (focal loss by default).")
    parser.add_argument("--weighted_loss", action="store_true", default=False, help="Enable the weighted bcewithlogits loss. Only works if the custom loss is enabled.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable fp16 mixed precision training.")
    parser.add_argument("--eval_metric", type=str, default="f1_micro", choices=[
        'loss', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
        'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples',
        'matthews_macro', 'matthews_micro',
        'roc_auc_micro', 'roc_auc_macro', 'roc_auc_weighted', 'roc_auc_samples',
        'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
        'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
        'hamming_loss', 'accuracy', 'ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10'],
        help="Evaluation metric to use on the validation set.")
    parser.add_argument("--full_metrics", action="store_true", default=False, help="Compute all the metrics during the evaluation.")
    parser.add_argument("--trust_remote", action="store_true", default=False, help="Trust the remote code for the model.")
    parser.add_argument("--save_class_report", action="store_true", default=False, help="Save the classification report.")
    parser.add_argument("--class_report_step", type=int, default=1, help="Number of epochs before creating a new classification report.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--models_path", type=str, default="models/", help="Save path of the models.")

    data_path = f"{file_basepath}/data"
    models_path = f"{file_basepath}/models"

    try:
        shutil.rmtree(data_path)
    except:
        print("Error deleting files")                       
    try:
        train_data.download(f"{file_basepath}/")
    except:
        print("Error downloading files")                       

    args = [
        "--lang", lang,
        "--seeds", seeds,
        "--device", device,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--max_grad_norm", str(max_grad_norm),
        "--threshold", str(threshold),
        "--eval_metric", eval_metric,
        "--class_report_step", str(class_report_step),
        "--data_path", data_path,
        "--models_path", models_path
    ]
    if custom_loss: args.append("--custom_loss")
    if weighted_loss: args.append("--weighted_loss")
    if fp16: args.append("--fp16")
    if full_metrics: args.append("--full_metrics")
    if trust_remote: args.append("--trust_remote")
    if save_class_report: args.append("--save_class_report")
        
    args = parser.parse_args(args)  

    res = start_train(args)

    # Load the seeds for the different splits
    if args.seeds != "all":
        seeds = args.seeds.split(",")
    else:
        seeds = [name.split("_")[1] for name in listdir(path.join(args.data_path, args.lang)) if "split" in name]
    for seed in seeds:    
        metrics = {}
        try: 
            metrics_path = path.join(models_path, lang, seed, "metrics.json")
            print(f"Reading metrics at {metrics_path} ...") 
            with open(metrics_path, 'r') as file:
                metrics = json.load(file)
        except:
            print("No metrics saved")                       
            
        project.log_model(
            name=f"eurovoc-classifier-{seed}",
            kind="huggingface",
            base_model="google-bert/bert-base-cased",
            metrics=metrics,
            source=res[seed],
        )             
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from transformers import (
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
import numpy as np
import random
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

################################################################################################
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def save_model_parts(model, output_dir):
    #model.embed_tokens.weight torch.Size([32000, 4096])
    torch.save(model.model.embed_tokens.weight, os.path.join(output_dir, "embed_tokens.pt"))
    #lm_head.weight torch.Size([32000, 4096])
    torch.save(model.lm_head.weight, os.path.join(output_dir, "lm_head.pt"))

################################################################################################
def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

################################################################################################
class Logger:
    def __init__(self, log_dir="logs", log_txt="train_log.txt"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_txt_path = os.path.join(log_dir, log_txt)
        self.writer = SummaryWriter(log_dir=log_dir)

        # clear old files, optional
        with open(self.log_txt_path, "w") as f:
            f.write("===== Training Log =====\n")

    def log(self, step, info: dict):
        """
        info: { 'loss': val, 'acc': val, ...}
        Can write: dictionary stat Like:
        {
        'overall_accuracy': overall_accuracy,
        'accuracy_per_class': accuracy_per_class,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        "f1_scores": f1_scores,
        'classification_report': report
        }

        write into txt and tensorboard
        """
        msg = f"[Step {step}] "
        for k,v in info.items():
            if isinstance(v, (int, float)):
                msg += f"{k}: {v:.4f}   "
                self.writer.add_scalar(k, v, step)
            elif isinstance(v, (dict)):
                for subk, subv in v.items():
                    if isinstance(subv, (dict)):
                        for subsubk, subsubv in subv.items():
                            if isinstance(subsubv, (dict)):
                                pass
                            else:
                                self.writer.add_scalar(k+'-'+str(subk)+'-'+str(subsubk), subsubv, step)
                    else:
                        msg += f"{k+'-'+str(subk)}: {str(subv)}   "
                        self.writer.add_scalar(k+'-'+str(subk), subv, step)
        #print(msg)
        with open(self.log_txt_path, "a") as f:
            f.write(msg+"\n")

    def close(self):
        self.writer.close()

########################################################################################
def save_checkpoint(model, optimizer, step, path):
    """
    Save: weights, optimizers
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at step {step} -> {path}")

def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Load checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"]
    print(f"Loaded checkpoint from {path}, start_step={start_step}")
    return start_step

from sklearn.metrics import confusion_matrix, classification_report

def evaluate_multiclass_metrics(preds: torch.Tensor, 
                                labels: torch.Tensor, 
                                num_classes: int):

    # class: [0, 1, 2, ..., num_classes-1]
    classes = list(range(num_classes))
    
    # acc
    overall_accuracy = (preds == labels).float().mean().item()
    
    # acc per class
    accuracy_per_class = {}
    for cls in classes:
        mask = (labels == cls)
        if mask.sum().item() == 0:
            accuracy_per_class[cls] = 0.0
        else:
            accuracy_per_class[cls] = (preds[mask] == labels[mask]).float().mean().item()
    
    # tensor 2 numpy 4 sklearn
    if preds.is_cuda:
        pred_np = preds.cpu().numpy()
        label_np = labels.cpu().numpy()
    else:
        pred_np = preds.numpy()
        label_np = labels.numpy()
    
    # CMat
    cm = confusion_matrix(label_np, pred_np, labels=classes)
    
    # classification_report
    report = classification_report(label_np, pred_np, labels=classes, output_dict=True, zero_division=0)
    
    # individual metrics
    per_class_metrics = {}
    for cls in classes:
        cls_key = str(cls)
        if cls_key in report:
            per_class_metrics[cls] = {
                'precision': report[cls_key]['precision'],
                'recall': report[cls_key]['recall'],
                'f1-score': report[cls_key]['f1-score'],
                'support': report[cls_key]['support']
            }
        else:
            per_class_metrics[cls] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0
            }
    
    f1_scores = {cls: per_class_metrics[cls]['f1-score'] for cls in classes}
    ##
    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_per_class': accuracy_per_class,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        "f1_scores": f1_scores,
        'classification_report': report
    }
    
    

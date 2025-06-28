import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_adni_img_gene_data_(root):
    data_list = []
    cases = [s for s in os.listdir(root) if not s.startswith('.')]
    for s in cases:
        patches = torch.Tensor(np.load(os.path.join(root, s,'pair.npy'))).float()
        labels = np.load(os.path.join(root, s,'label.npy')).astype(np.int16)
        label0 = labels[0]
        label1 = labels[1]
        gene = np.load(os.path.join(root, s,'gtype.npy'))
        gene =  torch.Tensor(gene.flatten()).float()
        
        data_list.append({"patches": patches, "label0": label0, "label1": label1, "gene": gene})
        
    return data_list

class CustomPatchDataset(Dataset):
    """
    sample:
      - patches: (156, 360)
      - label0: int in {0, 1, 2}
      - label1: int in {0, 1}

    add collate_fn, CLS1, CLS2 (140 + 2, 128)。
    """

    def __init__(self, data_list):
        """
        data_list: list of dict:
          {
            "patches": np.array of shape (156, 360),
            "label1": int,
            "label2": int
          }
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        patches = item["patches"]  # shape [156, 360]
        label0 = item["label0"]
        label1 = item["label1"]
        gene = item["gene"]
        return {
            "patches": patches, #Tensor
            "label0": label0, #int
            "label1": label1, #int
            "gene": gene #Tensor
        }

def collate_fn(batch, cls_dim=360):
    """
    batch: list of dict
    Add CLS token, label1/label2
    shape (156+2, 128)
    """
    # batch_size
    batch_size = len(batch)
    max_patches = len(batch[0]["patches"])  # 156

    # gird
    patch_tensors = []
    label1_list = []
    label0_list = []
    gene_list = []
    for i in range(batch_size):
        patches = batch[i]["patches"]
        label0 = batch[i]["label0"]
        label1 = batch[i]["label1"]
        genes = batch[i]["gene"]

        patch_tensors.append(patches)
        label0_list.append(label0)
        label1_list.append(label1)
        gene_list.append(genes)

    patch_tensors = torch.stack(patch_tensors)  # (B, 156, 360)
    label0_tensor = torch.tensor(label0_list, dtype=torch.long)
    label1_tensor = torch.tensor(label1_list, dtype=torch.long)
    gene_tensor = torch.stack(gene_list) # (B, 2 * 126)
    
    return {
        "patch_tensors": patch_tensors,  # [B, 156, 360]
        "label0": label0_tensor,         # [B]
        "label1": label1_tensor,          # [B]
        "genes": gene_tensor # [B, 2 * 126]
    }

def create_dataloader(data_list, batch_size, shuffle=True):
    dataset = CustomPatchDataset(data_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn)

if __name__ == "__main__":
    training_data = load_adni_img_gene_data_('/home/yan/Craftable/work2025/MICCAI_gene/Data/train')
    train_loader = create_dataloader(training_data, 1000)

    batch_ = next(iter(train_loader))
    # print(batch_["patch_tensors"])
    # print(batch_["label0"])
    # print(batch_["label1"])
    gene_tensor = batch_["genes"]
    print(gene_tensor.shape)
        
    
    # batch_ = next(iter(train_loader))
    # print(batch_["patch_tensors"])
    # print(batch_["genes"])
    # print(batch_["label0"])
    # print(batch_["genes"])
    
    test_data = load_adni_img_gene_data_('/home/yan/Craftable/work2025/MICCAI_gene/Data/test')
    test_loader = create_dataloader(test_data, 1000)
    
    batch_ = next(iter(test_loader))
    # print(batch_["patch_tensors"])
    # print(batch_["label0"])
    # print(batch_["label1"])
    gene_tensor = batch_["genes"]
    print(gene_tensor.shape)
        
    
    
import torch
from sklearn.metrics import accuracy_score
import copy
def test_model(model, test_dataloader, losser, device, Isave_feature=False):
    model.eval()
    true_labels = []
    pred_labels = []
    test_loss = 0
    raw_features = []
    DB_features = []
    fusion_features = []
    informer_features = []
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs = inputs.type(torch.FloatTensor).to(device)
            target = target.type(torch.LongTensor).to(device)
            inputs, target = inputs.to(device), target.to(device)
            raw_features.append(inputs)
            outputs, DB_out, fusion_out, informer_out = model(inputs)
            DB_features.append(DB_out)
            fusion_features.append(fusion_out)
            informer_features.append(informer_out)
            loss = losser(outputs, target)
            test_loss += loss.item()
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    test_loss /= len(test_dataloader)
    accuracy = accuracy_score(true_labels, pred_labels)
    raw_features_npy = torch.cat(raw_features, dim=0).cpu().numpy()
    DB_features_npy = torch.cat(DB_features, dim=0).cpu().numpy()
    fusion_features_npy = torch.cat(fusion_features, dim=0).cpu().numpy()
    informer_features_nupy = torch.cat(informer_features, dim=0).cpu().numpy()
    if Isave_feature:
        return raw_features_npy, DB_features_npy, fusion_features_npy, informer_features_nupy
    else:
        return accuracy, test_loss, true_labels, pred_labels


def test_saveweight(model, test_dataloader, losser, device, Isave_weight=False):
    model.eval()
    test_loss = 0
    all_attn_weight = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs = inputs.type(torch.FloatTensor).to(device)
            target = target.type(torch.LongTensor).to(device)
            inputs, target = inputs.to(device), target.to(device)
            outputs, attn_weight = model(inputs)
            all_attn_weight.extend(attn_weight)
            loss = losser(outputs, target)
            test_loss += loss.item()
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    test_loss /= len(test_dataloader)
    accuracy = accuracy_score(true_labels, pred_labels)
    all_attn_weight_npy = torch.cat(all_attn_weight, dim=0).cpu().numpy()
    return accuracy, test_loss, all_attn_weight_npy

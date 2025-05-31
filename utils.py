import json
import csv
import torch
import pandas as pd

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(
            data, f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
        )

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(data, file_path, quoting=csv.QUOTE_ALL):
    data.to_csv(file_path, index=False, encoding='utf-8', quoting=quoting)

def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

def save_tsv(data, file_path, quoting=csv.QUOTE_ALL):
    data.to_csv(file_path, sep='\t', index=False, encoding='utf-8', quoting=quoting)

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_txt_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()

def write_txt(content, file_path, encoding="utf-8"):
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)

def write_txt_lines(lines, file_path, encoding="utf-8"):
    # lines (list):
    with open(file_path, "w", encoding=encoding) as f:
        f.writelines(lines)


# ************************************************************
def weight_merging_gaussian(model_1, model_2, ptw=None, lamda=2):
    merged = {}
    for k in model_1.keys():
        if k not in ptw:
            merged[k] = model_1[k]
        else:
            delta_model1 = model_1[k] - ptw[k]
            delta_model2 = model_2[k] - ptw[k]

            mu1, sigma1 = delta_model1.mean(), delta_model1.std()
            mu2, sigma2 = delta_model2.mean(), delta_model2.std()

            mu_avg = (mu1 + mu2) / 2
            sigma_est = min(sigma1, sigma2)

            within_mask1 = (delta_model1 > mu_avg - lamda * sigma_est) & (delta_model1 < mu_avg + lamda * sigma_est)
            model_1k_within = delta_model1 * within_mask1
            model_1k_outlier = delta_model1 * (~within_mask1)

            within_mask2 = (delta_model2 > mu_avg - lamda * sigma_est) & (delta_model2 < mu_avg + lamda * sigma_est)
            model_2k_within = delta_model2 * within_mask2
            model_2k_outlier = delta_model2 * (~within_mask2)

            merged[k] = ptw[k] + 1. * (model_1k_outlier + model_2k_outlier) + 0.5 * (model_1k_within + model_2k_within)
            # merged[k] = ptw[k] + 1.0 * (model_1k_within + model_2k_within)

    return merged

def weight_merging_fnorm(model_1, model_2, ptw=None):
    merged = {}
    for k in model_1.keys():
        if k not in ptw:
            merged[k] = model_1[k]
        else:
            delta_model1 = model_1[k] - ptw[k]
            delta_model2 = model_2[k] - ptw[k]

            if delta_model1.dim() == 1:
                scale1 = 1. / (torch.norm(delta_model1, p=2))
                scale2 = 1. / (torch.norm(delta_model2, p=2))
            elif delta_model1.dim() == 2:
                scale1 = 1. / (torch.norm(delta_model1, p='fro').mean())
                scale2 = 1. / (torch.norm(delta_model2, p='fro').mean())
            else:
                raise ValueError(f"delta_model1.dim() = {delta_model1.dim()}")

            weighted_sum = scale1 * delta_model1 + scale2 * delta_model2
            weighted_norm  = weighted_sum / (scale1 + scale2)

            merged[k] = ptw[k] + weighted_norm
    return merged

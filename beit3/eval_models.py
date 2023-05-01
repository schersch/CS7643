import json
import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------
# Begin original code contribution
# ---------------------------------

ensembles = {
    "base_single_unaug": [{"size": "base", "seed": 42, "data": "unaug", "lr": ""}],
    "base_single_aug": [{"size": "base", "seed": 42, "data": "aug", "lr": ""}],
    "large_single_unaug": [{"size": "large", "seed": 51, "data": "unaug", "lr": ""}],
    "large_single_unaug_best": [{"size": "large", "seed": 51, "data": "unaug", "lr": "lr_"}],
    "large_single_aug": [{"size": "large", "seed": 51, "data": "aug", "lr": ""}],
    "large_ensemble_unaug": [
        {"size": "large", "seed": 47, "data": "unaug", "lr": ""},
        {"size": "large", "seed": 48, "data": "unaug", "lr": ""},
        {"size": "large", "seed": 49, "data": "unaug", "lr": ""},
        {"size": "large", "seed": 50, "data": "unaug", "lr": ""},
        {"size": "large", "seed": 51, "data": "unaug", "lr": ""},
    ],
    "large_ensemble_aug": [
        {"size": "large", "seed": 47, "data": "aug", "lr": ""},
        {"size": "large", "seed": 48, "data": "aug", "lr": ""},
        {"size": "large", "seed": 49, "data": "aug", "lr": ""},
        {"size": "large", "seed": 50, "data": "aug", "lr": ""},
        {"size": "large", "seed": 51, "data": "aug", "lr": ""},
    ],
    "lr_9e-5": [{"size": "large", "seed": 53, "data": "unaug", "lr": ""}],
    "lr_3e-5": [{"size": "large", "seed": 54, "data": "unaug", "lr": ""}],
    "decay_0.1": [{"size": "large", "seed": 55, "data": "unaug", "lr": ""}],
    "decay_0.3": [{"size": "large", "seed": 56, "data": "unaug", "lr": ""}],
    "decay_0.4": [{"size": "large", "seed": 57, "data": "unaug", "lr": ""}],
    "mmt_0.7": [{"size": "large", "seed": 58, "data": "unaug", "lr": ""}],
    "mmt_0.5": [{"size": "large", "seed": 59, "data": "unaug", "lr": ""}],
    "large_ensemble_unaug_lr": [
        {"size": "large", "seed": 47, "data": "unaug", "lr": "lr_"},
        {"size": "large", "seed": 48, "data": "unaug", "lr": "lr_"},
        {"size": "large", "seed": 49, "data": "unaug", "lr": "lr_"},
        {"size": "large", "seed": 50, "data": "unaug", "lr": "lr_"},
        {"size": "large", "seed": 51, "data": "unaug", "lr": "lr_"},
    ],
    "large_ensemble_aug_lr": [
        {"size": "large", "seed": 47, "data": "aug", "lr": "lr_"},
        {"size": "large", "seed": 48, "data": "aug", "lr": "lr_"},
        {"size": "large", "seed": 49, "data": "aug", "lr": "lr_"},
        {"size": "large", "seed": 50, "data": "aug", "lr": "lr_"},
        {"size": "large", "seed": 51, "data": "aug", "lr": "lr_"},
    ],
}


def eval_ensemble(models):
    results = []
    for model in models:
        seed = model["seed"]
        size = model["size"]
        data = model["data"]
        lr = model["lr"]
        results_path = f"results/beit3_{seed}_{size}_{data}_{lr}preds.json"
        with open(results_path) as f:
            results.append(json.load(f))
    avg_probs = np.mean([np.array(result["probs"]) for result in results], axis=0)
    preds = (avg_probs > 0.5).astype(int)
    acc = np.mean(np.array(results[0]["labels"]) == preds)
    auroc = roc_auc_score(results[0]["labels"], avg_probs)
    return acc, auroc


if __name__ == "__main__":
    final_results = {}
    for name, models in ensembles.items():
        acc, auroc = eval_ensemble(models)
        print(f"{name}: final test acc {acc:.4f}, final test auroc {auroc:.4f}")
        final_results[name] = {"acc": f"{acc:.4f}", "auroc": f"{auroc:.4f}"}
    with open("results/final_results.json", "w") as f:
        f.write(json.dumps(final_results))

# -------------------------------
# End original code contribution
# -------------------------------

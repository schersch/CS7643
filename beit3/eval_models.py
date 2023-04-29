import json
import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------
# Begin original code contribution
# ---------------------------------

ensembles = {
    "base_single_unaug": [{"size": "base", "seed": 42, "data": "unaug"}],
    "base_single_aug": [{"size": "base", "seed": 42, "data": "aug"}],
    "large_single_unaug": [{"size": "large", "seed": 51, "data": "unaug"}],
    "large_single_aug": [{"size": "large", "seed": 51, "data": "aug"}],
    "large_ensemble_unaug": [
        {"size": "large", "seed": 47, "data": "unaug"},
        {"size": "large", "seed": 48, "data": "unaug"},
        {"size": "large", "seed": 49, "data": "unaug"},
        {"size": "large", "seed": 50, "data": "unaug"},
        {"size": "large", "seed": 51, "data": "unaug"},
    ],
    "large_ensemble_aug": [
        {"size": "large", "seed": 47, "data": "aug"},
        {"size": "large", "seed": 48, "data": "aug"},
        {"size": "large", "seed": 49, "data": "aug"},
        {"size": "large", "seed": 50, "data": "aug"},
        {"size": "large", "seed": 51, "data": "aug"},
    ],
}


def eval_ensemble(models):
    results = []
    for model in models:
        seed = model["seed"]
        size = model["size"]
        data = model["data"]
        results_path = f"results/beit3_{seed}_{size}_{data}_preds.json"
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

import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------
# Begin original code contribution
# ---------------------------------

def graph_auroc_results():
    models = [
        ("Base Orig.", "beit3_42_base_unaug", "val_auroc"),
        ("Base Aug.", "beit3_42_base_aug", "val_auroc"),
        ("Large Orig.", "beit3_51_large_unaug", "val_auroc"),
        ("Large Aug.", "beit3_51_large_aug", "val_auroc"),
    ]
    stats = []
    for model in models:
        with open(f"models/finetuned/{model[1]}/log.txt", "r") as f:
            stats.append([json.loads(l)[model[2]] for l in f])

    plt.figure()
    for i, stat_list in enumerate(stats):
        plt.plot(stat_list, label=models[i][0])

    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=0)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"graphs/beit3_auroc_vs_epoch.png")
    plt.close()


def graph_loss_results():
    models = [
        ("Base Train Loss", "beit3_42_base_unaug", "train_loss"),
        ("Base Val Loss", "beit3_42_base_unaug", "val_loss"),
        ("Large Train Loss", "beit3_51_large_unaug", "train_loss"),
        ("Large Val Loss", "beit3_51_large_unaug", "val_loss"),
    ]
    stats = []
    for model in models:
        with open(f"models/finetuned/{model[1]}/log.txt", "r") as f:
            stats.append([json.loads(l)[model[2]] for l in f])

    plt.figure()
    for i, stat_list in enumerate(stats):
        plt.plot(stat_list, label=models[i][0])

    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=0)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"graphs/beit3_loss_vs_epoch.png")
    plt.close()


if __name__ == "__main__":
    graph_loss_results()
    graph_auroc_results()


# ---------------------------------
# End original code contribution
# ---------------------------------

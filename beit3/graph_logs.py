import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------
# Begin original code contribution
# ---------------------------------

models = [
    ("Base, unaug", "beit3_42_base_unaug"),
    ("Base, aug", "beit3_42_base_aug"),
    ("Large, unaug", "beit3_51_large_unaug"),
    ("Large, aug", "beit3_51_large_aug"),
]


def graph_results(models, field, metric_name):
    stats = []
    for model in models:
        with open(f"models/finetuned/{model[1]}/log.txt", "r") as f:
            stats.append([json.loads(l)[field] for l in f])

    plt.figure()
    for i, stat_list in enumerate(stats):
        plt.plot(stat_list, label=models[i][0])

    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=0)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.savefig(f"graphs/beit3_{field}_vs_epoch.png")
    plt.close()


if __name__ == "__main__":
    graph_params = [[models, "train_loss", "Loss"], [models, "val_loss", "Loss"], [models, "val_auroc", "AUROC"], [models, "val_acc", "Accuracy"]]
    for params in graph_params:
        graph_results(*params)

# ---------------------------------
# End original code contribution
# ---------------------------------

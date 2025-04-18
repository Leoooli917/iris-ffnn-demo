import numpy as np
import matplotlib.pyplot as plt
from neural_networks.utils import AttrDict
from neural_networks.models import initialize_model
from neural_networks.datasets import initialize_dataset
from neural_networks.layers import initialize_layer
import pickle

np.random.seed(189)

grid = [
    {"lr": 0.1,   "hidden": 10},
    {"lr": 0.01,  "hidden": 25},
    {"lr": 0.001, "hidden": 50},
]

batch_size = 25
epochs     = 50

results = []

for run_i, params in enumerate(grid):
    print(f"\n=== Run {run_i+1}/{len(grid)}: lr={params['lr']}, hidden={params['hidden']} ===")

    fc1 = AttrDict({
        "name":        "fully_connected",
        "activation":  "relu",
        "weight_init": "xavier_uniform",
        "n_out":       params["hidden"],
    })
    fc_out = AttrDict({
        "name":        "fully_connected",
        "activation":  "softmax",
        "weight_init": "xavier_uniform",
        "n_out":       None,      
    })
    layer_args = [fc1, fc_out]

    optimizer_args = AttrDict({
        "name":       "SGD",
        "lr":         params["lr"],
        "lr_scheduler":"constant",
        "lr_decay":   0.99,
        "stage_length": None,
        "staircase":  True,
        "clip_norm":  1.0,
        "momentum":   0.9,
    })

    model_args = AttrDict({
        "name":          "feed_forward",
        "loss":          "cross_entropy",
        "layer_args":    layer_args,
        "optimizer_args":optimizer_args,
        "seed":          0,
    })
    model = initialize_model(
        name           = model_args.name,
        loss           = model_args.loss,
        layer_args     = model_args.layer_args,
        optimizer_args = model_args.optimizer_args,
        logger         = None   
    )

    dataset = initialize_dataset(name="iris", batch_size=batch_size)

    fc_out["n_out"] = dataset.out_dim
    output_layer = initialize_layer(**fc_out)
    model.layers.append(output_layer)

    train_losses = []
    for epoch in range(epochs):
        for _ in range(dataset.train.samples_per_epoch):
            X, Y    = dataset.train.sample()
            Y_hat   = model.forward(X)
            loss    = model.backward(Y, Y_hat)
            model.update(epoch)
            train_losses.append(loss)

    test_log = model.test(dataset, save_predictions=False)
    test_error = np.mean(test_log["error"])
    test_acc   = 1 - test_error

    print(f"--> Final test accuracy: {test_acc:.4f}")

    results.append({
        "lr":           params["lr"],
        "hidden":       params["hidden"],
        "loss_curve":   train_losses,
        "test_accuracy":test_acc,
    })

best = max(results, key=lambda r: r["test_accuracy"])
print("\n=== Summary of runs ===")
for r in results:
    print(f"lr={r['lr']:<8} hidden={r['hidden']:<3}  test_acc={r['test_accuracy']:.4f}")
print(f"\nBest setting: lr={best['lr']}, hidden={best['hidden']}  → test_acc={best['test_accuracy']:.4f}")

plt.figure(figsize=(8,5))
plt.plot(best["loss_curve"], label=f"lr={best['lr']}, hidden={best['hidden']}")
plt.xlabel("Minibatch iteration")
plt.ylabel("Training loss")
plt.title("Training loss curve for best Iris run")
plt.legend()
plt.tight_layout()
plt.show()

with open("iris_ffnn.pkl","wb") as f:
    pickle.dump(model.state_dict(), f)


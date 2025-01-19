import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

df = pd.read_csv('sp500.csv')
df = df.set_index('Date')
data = torch.from_numpy(df.to_numpy()).to(torch.float32)
days_in_quarter = 64
num_quarters = data.size(0) // days_in_quarter
num_days = num_quarters * days_in_quarter
data = data[:num_days]
train_days = int(0.8 * num_quarters) * days_in_quarter
train_stds = data[:train_days].std(dim=0)
train_means = data[:train_days].mean(dim=0)

# Plot regression results on test set
def plot_returns_regression(configs, labels, fig_name, stock_name, preds):
    stock_index = stock_lookup[stock_name]
    plt.clf()
    plt.figure(figsize=(10, 6))
    colors = [(0.650, 0.120, 0.240, 0.6),  # red
              (0.122, 0.467, 0.706, 0.6), # blue
              (1.000, 0.498, 0.055), # orange
              (0.580, 0.403, 0.741, 0.6), # purple
              ]
    plt.rc('axes', prop_cycle=cycler('color', colors))

    for (gnn, use_spatial, corr_name, corr_scope), label in zip(configs, labels):
        eval_dataset = get_dataset(corr_name, corr_scope)['test_samples']
        model = get_model(gnn, use_spatial, corr_name, corr_scope, load_weights=True)
        if model is None:
            continue
        model.eval()
        with torch.no_grad():
            if label not in preds:
                y_hats = list(map(lambda snapshot: infer(model, snapshot).squeeze().cpu(), eval_dataset))
                preds[label] = y_hats
            else:
                y_hats = preds[label]
            ys = [snapshot.y.cpu() for snapshot in eval_dataset]
            targets = ys
            assert(len(y_hats) == len(ys))
            x = np.array(range(len(ys) - 1))
            print(f'{label} Overall MAE: {mae(torch.stack(y_hats, dim=1), torch.stack(ys, dim=1))}')
            ys = torch.tensor([y[stock_index] for y in ys])
            y_hats = torch.tensor([y_hat[stock_index] for y_hat in y_hats])
            print(f'{label} Stock {stock_index} MAE: {mae(y_hats, ys)}')

            pred_price = y_hats * train_stds[stock_index] + train_means[stock_index]
            pred_return = (pred_price[1:] - pred_price[:-1]) / pred_price[:-1]
            plt.plot(x, pred_return, label=label, linewidth=1)

    real_price = ys * train_stds[stock_index] + train_means[stock_index]
    real_return = (real_price[1:] - real_price[:-1]) / real_price[:-1]
    plt.plot(x, real_return, label="Real", color='green')
    plt.legend(fontsize=14)
    plt.title(f'Predicted vs Real {stock_name} Stock Daily Returns', fontsize=20)
    plt.xlabel('Days', fontsize=16)
    plt.ylabel('Daily Return', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.savefig(fig_name)
    plt.show()
    return preds, targets
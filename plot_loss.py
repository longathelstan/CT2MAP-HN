import re
import matplotlib.pyplot as plt

log_file = '/data/lowngworkspace/CT2MAP-HN/train_baseline.log'
out_png = '/home/long/.gemini/antigravity-ide/brain/6343d00c-04af-4035-b197-b6389c3528ed/loss_curve.png'

epochs_dict = {}
with open(log_file, 'r') as f:
    for line in f:
        match_train_val = re.search(r'Epoch (\d+)/\d+\s+train_loss=([0-9.nan]+)\s+val_loss=([0-9.nan]+)', line)
        if match_train_val:
            ep = int(match_train_val.group(1))
            t_loss_str = match_train_val.group(2)
            v_loss_str = match_train_val.group(3)
            
            t_loss = float(t_loss_str) if t_loss_str != 'nan' else None
            v_loss = float(v_loss_str) if v_loss_str != 'nan' else None
            epochs_dict[ep] = {'train': t_loss, 'val': v_loss}

sorted_eps = sorted(epochs_dict.keys())

# Filter out early NaN epochs from the plot if any
train_epochs = [ep for ep in sorted_eps if epochs_dict[ep]['train'] is not None]
train_losses = [epochs_dict[ep]['train'] for ep in train_epochs]

# Validation loss is only meaningful every val_interval (5)
val_epochs = [ep for ep in sorted_eps if epochs_dict[ep]['val'] is not None and ep % 5 == 4] # wait, if val interval is 5, it might be at epoch 4, 9, 14. Actually let's just plot all non-None val losses.
val_epochs = []
val_losses = []
for ep in sorted_eps:
    v = epochs_dict[ep]['val']
    if v is not None and v > 0: # filter out 0.000 if any
        val_epochs.append(ep)
        val_losses.append(v)

plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5, alpha=0.8)
plt.plot(val_epochs, val_losses, label='Val Loss', marker='o', color='#ff7f0e', linewidth=2, markersize=5)

# Mark the best model
best_val = min(val_losses)
best_ep = val_epochs[val_losses.index(best_val)]
plt.axvline(x=best_ep, color='green', linestyle='--', alpha=0.5, label=f'Best Val Loss ({best_val:.4f} at Ep {best_ep})')
plt.scatter([best_ep], [best_val], color='red', s=100, zorder=5)

plt.title('Training and Validation Loss Curve (Baseline P40)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Plot saved to {out_png}")

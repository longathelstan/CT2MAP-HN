# -*- coding: utf-8 -*-
"""Generate training progress report as a Markdown table."""
import re
import platform
from pathlib import Path

def main():
    log_path = Path("train_baseline.log")
    
    # Path routing based on OS (running in WSL vs Windows)
    if platform.system() == "Linux":
        artifact_dir = Path("/mnt/c/Users/Long/.gemini/antigravity-ide/brain/98bb5754-867d-4dc3-b556-3e0c7a3d2874")
    else:
        artifact_dir = Path("C:/Users/Long/.gemini/antigravity-ide/brain/98bb5754-867d-4dc3-b556-3e0c7a3d2874")
        
    artifact_path = artifact_dir / "training_status.md"
    
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return
        
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    
    # Find all completed epoch log lines
    # Format: 2026-07-02 09:44:11 [INFO] src.train.engine: Epoch 1/100  train_loss=1.62229  val_loss=1.61232  lr=1.00e-04  time=216.4s
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] src\.train\.engine: Epoch (\d+)/100\s+train_loss=([\d\.]+)\s+val_loss=([\d\.]+)\s+lr=([\d\.e\-]+)\s+time=([\d\.]+)"
    matches = re.findall(pattern, text)
    
    # Check if the training task is still active in the log
    is_active = False
    last_epoch_active = None
    pct_active = None
    
    tqdm_matches = re.findall(r"Train Epoch (\d+):\s+(\d+)%\|", text)
    val_tqdm_matches = re.findall(r"Validate:\s+(\d+)%\|", text)
    
    if tqdm_matches:
        last_epoch_active = int(tqdm_matches[-1][0])
        pct_active = int(tqdm_matches[-1][1])
        is_active = True
        
    if "Training complete!" in text or "Early stopping triggered" in text:
        is_active = False
        
    status_str = "🟢 ACTIVE" if is_active else "🔴 STOPPED/COMPLETED"
    
    # Build markdown content
    md = []
    md.append("# CT2MAP-HN Baseline Training Progress Report\n")
    md.append(f"**Current Status:** {status_str}\n")
    
    if is_active and last_epoch_active is not None:
        md.append(f"* **Active Epoch:** Epoch {last_epoch_active} (Progress: {pct_active}%)\n")
        
    md.append("## Training History\n")
    md.append("| Epoch | Timestamp | Train Loss | Val Loss | Learning Rate | Epoch Time (s) |")
    md.append("|---|---|---|---|---|---|")
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    for ts, ep, t_loss, v_loss, lr, duration in matches:
        epoch_num = int(ep)
        val_loss_val = float(v_loss)
        
        # Highlight best validation loss
        val_cell = f"**{v_loss}** 🏆" if val_loss_val < best_val_loss else v_loss
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = epoch_num
            
        md.append(f"| {epoch_num} | {ts} | {t_loss} | {val_cell} | {lr} | {duration} |")
        
    md.append("\n## Highlights\n")
    if best_epoch != -1:
        md.append(f"* **Best Validation Loss:** `{best_val_loss:.6f}` at **Epoch {best_epoch}**\n")
        
    checkpoint_dir = Path("checkpoints")
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if best_checkpoint.exists():
        md.append(f"* **Best Checkpoint:** [{best_checkpoint.name}](file:///{best_checkpoint.resolve().as_posix()}) (Size: {best_checkpoint.stat().st_size / (1024*1024):.1f} MB)\n")
    else:
        md.append(f"* **Best Checkpoint:** Not saved yet\n")
        
    # Write to artifact directory
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Report successfully generated at: {artifact_path}")

if __name__ == "__main__":
    main()

import csv
import os
import time
import torch
from tqdm import tqdm

def save_latest(epoch, best_val_loss, model, optimizer, scheduler, checkpoint_dir):

    path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print("Saved latest epoch.")

def save_best(model, checkpoint_dir):

    path = os.path.join(checkpoint_dir, "best.pth")
    torch.save(model.state_dict(), path)
    print("Saved best epoch.")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):

    checkpoint = torch.load(checkpoint_dir)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"Resuming from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], checkpoint['best_val_loss']

def log_epoch(checkpoint_dir, row):

    path = os.path.join(checkpoint_dir, "history.csv")
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds", "is_best"])
        w.writerow(row)

def train_model(model, train_loader, val_loader, device, optimizer, criterion, scheduler, run_name, epochs=10, resume=None):

    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    if resume is not None:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1} Complete.")
        print(f"Average Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                output = model(images)

                val_loss += criterion(output, masks).item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best(model, checkpoint_dir)

        save_latest(epoch + 1, best_val_loss, model, optimizer, scheduler, checkpoint_dir)

        log_epoch(checkpoint_dir, [
            epoch + 1,
            round(epoch_loss, 6),
            round(val_loss, 6),
            current_lr,
            round(time.time() - epoch_start, 1),
            int(val_loss < best_val_loss),
        ])

    print('DONE!')
    return model
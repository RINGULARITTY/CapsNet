from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn.functional as F
import torch
from capsnet.utils import margin_loss

def train_capsnet(model, train_loader, val_loader, device, epochs=25, alpha=0.0005, lr=1e-3, progress=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    f1_scores = []

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                probs, recon = model(data, target)
                loss_m = margin_loss(probs, target)
                loss_r = F.mse_loss(recon, data)
                loss = loss_m + alpha * loss_r
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                with torch.amp.autocast("cuda"):
                    probs, recon = model(data, target)
                    loss_m = margin_loss(probs, target)
                    loss_r = F.mse_loss(recon, data)
                    loss = loss_m + alpha * loss_r
                val_loss += loss.item() * data.size(0)
                preds = probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_scores.append(f1_macro)
        scheduler.step(val_loss)
        
        if progress:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val F1: {f1_macro*100:.2f}%")
        
        if len(f1_scores) >= 3 and sum(f1_scores) <= 1.5: # leaves early if the model is too bad
            return model, 0

    top3 = sorted(f1_scores, reverse=True)[:3]
    avg_top3 = sum(top3) / len(top3)
    
    if progress:
        print(f"Avg Top 3 F1: {avg_top3*100:.2f}%")
    
    return model, avg_top3
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import assd


def iou_score(y_true, y_pred, epsilon=1e-8):
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()


def gradient_penalty(discriminator, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Refine Mask (unchanged)
def refine_mask(masks, device, threshold=0.3, min_area=300):
    masks = (masks > threshold).float()
    masks = (masks.detach().cpu().numpy() * 255).astype(np.uint8)
    masks = masks.squeeze(1)
    refined_masks = []
    for mask in masks:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 255
        refined_masks.append(filtered_mask)
    refined_masks = np.stack(refined_masks)[:, np.newaxis, :, :]
    return torch.from_numpy(refined_masks / 255.0).float().to(device)

# Enhanced Evaluation Function
def evaluate(model, loader, device):
    model.eval()
    dice_scores, iou_scores, hd_scores, precisions, recalls, assd_scores = [], [], [], [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            pred_masks = model(images)
            refined_masks = refine_mask(pred_masks, device)
            # Dice
            dice = 2 * torch.sum(refined_masks * masks, dim=(1, 2, 3)) / (refined_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-8)
            dice_scores.extend(dice.cpu().numpy())
            # IoU
            iou = iou_score(masks, refined_masks)
            iou_scores.append(iou.item())
            # Hausdorff Distance
            hd_mean, _ = hausdorff_distance(masks, refined_masks)
            hd_scores.append(hd_mean)
            # Precision and Recall
            prec, rec = precision_recall(masks, refined_masks)
            precisions.append(prec)
            recalls.append(rec)
            # ASSD
            assd_mean, _ = avg_symmetric_surface_distance(masks, refined_masks)
            assd_scores.append(assd_mean)
    return (np.mean(dice_scores), np.std(dice_scores),
            np.mean(iou_scores), np.std(iou_scores),
            np.mean(hd_scores), np.std(hd_scores),
            np.mean(precisions), np.std(precisions),
            np.mean(recalls), np.std(recalls),
            np.mean(assd_scores), np.std(assd_scores))

def hausdorff_distance(y_true, y_pred):
    hd_scores = []
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    for t, p in zip(y_true, y_pred):
        t_points = np.argwhere(t.squeeze())
        p_points = np.argwhere(p.squeeze())
        if len(t_points) == 0 or len(p_points) == 0:
            hd_scores.append(0.0)
        else:
            hd = max(directed_hausdorff(t_points, p_points)[0], directed_hausdorff(p_points, t_points)[0])
            hd_scores.append(hd)
    return np.mean(hd_scores), np.std(hd_scores)

def precision_recall(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = y_true.flatten(1), y_pred.flatten(1)
    tp = torch.sum(y_true * y_pred, dim=1)
    fp = torch.sum(y_pred * (1 - y_true), dim=1)
    fn = torch.sum(y_true * (1 - y_pred), dim=1)
    precision = (tp / (tp + fp + epsilon)).mean().item()
    recall = (tp / (tp + fn + epsilon)).mean().item()
    return precision, recall

def avg_symmetric_surface_distance(y_true, y_pred):
    assd_scores = []
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    for t, p in zip(y_true, y_pred):
        if t.sum() == 0 or p.sum() == 0:
            assd_scores.append(0.0)
        else:
            assd_scores.append(assd(t.squeeze(), p.squeeze()))
    return np.mean(assd_scores), np.std(assd_scores)

# Visualization Functions
def visualize_predictions(generator, loader, device, num_samples=3, init_type="CustomInit", plot_dir="plots"):
    generator.eval()
    images, masks = next(iter(loader))
    images, masks = images[:num_samples].to(device), masks[:num_samples].to(device)
    with torch.no_grad():
        pred_masks = generator(images)
        refined_masks = refine_mask(pred_masks, device)
    # mean, std = torch.tensor(full_dataset.mean).view(1, 3, 1, 1).to(device), torch.tensor(full_dataset.std).view(1, 3, 1, 1).to(device)
    # images_denorm = (images * std + mean).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    images_denorm = images #update
    plt.figure(figsize=(5 * num_samples, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images_denorm[i])
        plt.contour(masks[i].cpu().squeeze(), colors='green', linewidths=1)
        plt.contour(refined_masks[i].cpu().squeeze(), colors='red', linewidths=1)
        plt.title(f"Sample {i+1}", fontsize=12)
        plt.axis('off')
    plt.legend(['Ground Truth', 'Predicted'], loc='upper right', fontsize=10)
    plt.savefig(os.path.join(plot_dir, f"Predictions_{init_type}_c1.png"), dpi=500, bbox_inches='tight')
    plt.close()

def plot_test_metrics(test_metrics, init_type="CustomInit", plot_dir="plots"):
    labels = ['Dice', 'IoU', 'HD', 'Precision', 'Recall', 'ASSD']
    means = [test_metrics[i] for i in [0, 2, 4, 6, 8, 10]]
    stds = [test_metrics[i] for i in [1, 3, 5, 7, 9, 11]]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([np.random.normal(m, s, 100) for m, s in zip(means[:2] + means[3:5], stds[:2] + stds[3:5])], 
                labels=[labels[i] for i in [0, 1, 3, 4]])
    plt.title('Test Set Performance Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plot_dir, f"TestMetrics_{init_type}_c1.png"), dpi=500, bbox_inches='tight')
    plt.close()

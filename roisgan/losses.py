import torch
import torch.nn as nn

# Loss Functions 
def dice_loss(y_true, y_pred, epsilon=1e-8):
    numerator = 2 * torch.sum(y_true * y_pred, dim=(1, 2, 3))
    denominator = torch.sum(y_true + y_pred, dim=(1, 2, 3)) + epsilon
    return 1 - (numerator / denominator).mean()

def generator_loss(y_true, y_pred, fake_output):
    seg_loss = dice_loss(y_true, y_pred)
    adv_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
    return seg_loss + 0.1 * adv_loss

def discriminator_loss(real_output, fake_output, ground_truth):
    dice_real = dice_loss(ground_truth, real_output)
    bce_real = nn.BCELoss()(real_output, torch.ones_like(real_output))
    dice_fake = dice_loss(ground_truth, fake_output)
    bce_fake = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    return (dice_real + bce_real + dice_fake + bce_fake) / 4
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from roisgan.losses import generator_loss, discriminator_loss
from roisgan.utils import visualize_predictions, evaluate, plot_training_curves, plot_test_metrics
# Training Function
def train(generator, discriminator, train_loader, val_loader, test_loader, epochs, device, init_type="CustomInit", plot_dir="plots"):
    gen_optimizer = optim.Adam(generator.parameters(), lr=5e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5)
    best_val_dice = 0.0
    patience = 50
    epochs_no_improve = 0

    gen_losses, disc_losses, val_metrics = [], [], [[], [], [], [], [], []]  # Dice, IoU, HD, Prec, Rec, ASSD

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} ({init_type})")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            num_batches += 1

            gen_optimizer.zero_grad()
            pred_masks = generator(images)
            fake_output = discriminator(pred_masks)
            gen_loss = generator_loss(masks, pred_masks, fake_output)
            gen_loss.backward()
            gen_optimizer.step()

            disc_optimizer.zero_grad()
            real_output = discriminator(masks)
            fake_output = discriminator(pred_masks.detach())
            disc_loss = discriminator_loss(real_output, fake_output, masks) + 1.0 * gradient_penalty(discriminator, masks, pred_masks.detach(), device)
            disc_loss.backward()
            disc_optimizer.step()

            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()
            pbar.set_postfix({"Gen Loss": gen_loss.item(), "Disc Loss": disc_loss.item()})

        gen_losses.append(epoch_gen_loss / num_batches)
        disc_losses.append(epoch_disc_loss / num_batches)

        # Validation
        val_metrics_epoch = evaluate(generator, val_loader, device)
        val_metrics[0].append(val_metrics_epoch[0])  # Dice
        val_metrics[1].append(val_metrics_epoch[2])  # IoU
        val_metrics[2].append(val_metrics_epoch[4])  # HD
        val_metrics[3].append(val_metrics_epoch[6])  # Precision
        val_metrics[4].append(val_metrics_epoch[8])  # Recall
        val_metrics[5].append(val_metrics_epoch[10]) # ASSD
        print(f"Validation - Dice: {val_metrics_epoch[0]:.4f} ± {val_metrics_epoch[1]:.4f}, IoU: {val_metrics_epoch[2]:.4f} ± {val_metrics_epoch[3]:.4f}, "
              f"HD: {val_metrics_epoch[4]:.2f} ± {val_metrics_epoch[5]:.2f}, Prec: {val_metrics_epoch[6]:.4f} ± {val_metrics_epoch[7]:.4f}, "
              f"Rec: {val_metrics_epoch[8]:.4f} ± {val_metrics_epoch[9]:.4f}, ASSD: {val_metrics_epoch[10]:.2f} ± {val_metrics_epoch[11]:.2f}")

        if val_metrics_epoch[0] > best_val_dice:
            best_val_dice = val_metrics_epoch[0]
            torch.save(generator.state_dict(), f"{plot_dir}/apr29_best_generator_{init_type}_c1.pth")
            print(f"New best model saved with Val Dice: {best_val_dice:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        if epoch == 4 or epoch % 10 == 0:
            visualize_predictions(generator, val_loader, device, init_type=init_type, plot_dir=plot_dir)

    generator.load_state_dict(torch.load(f"{plot_dir}/apr29_best_generator_{init_type}_c1.pth"))
    test_metrics = evaluate(generator, test_loader, device)
    print(f"Test Set ({init_type}) - Dice: {test_metrics[0]:.4f} ± {test_metrics[1]:.4f}, IoU: {test_metrics[2]:.4f} ± {test_metrics[3]:.4f}, "
          f"HD: {test_metrics[4]:.2f} ± {test_metrics[5]:.2f}, Prec: {test_metrics[6]:.4f} ± {test_metrics[7]:.4f}, "
          f"Rec: {test_metrics[8]:.4f} ± {test_metrics[9]:.4f}, ASSD: {test_metrics[10]:.2f} ± {test_metrics[11]:.2f}")
    plot_test_metrics(test_metrics, init_type=init_type, plot_dir=plot_dir)
    visualize_predictions(generator, test_loader, device, init_type=init_type, plot_dir=plot_dir)

    return generator, test_metrics
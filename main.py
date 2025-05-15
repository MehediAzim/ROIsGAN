import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from roisgan.dataset import SegmentationDataset
from roisgan.models import Generator, Discriminator
from roisgan.train import train
from roisgan.config import CONFIG
import argparse
import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(CONFIG['plot_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train ROIsGAN for hippocampal segmentation.')
    parser.add_argument('--image_dir', default=CONFIG['image_dir'], help='Directory with input images.')
    parser.add_argument('--mask_dir', default=CONFIG['mask_dir'], help='Directory with mask images.')
    parser.add_argument('--use_custom_init', action='store_true', help='Use custom initialization for UNet.')
    args = parser.parse_args()

    # Update config with command-line arguments
    CONFIG['image_dir'] = args.image_dir
    CONFIG['mask_dir'] = args.mask_dir

    # Setup
    setup_logging()
    logging.info("Starting ROIsGAN training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CONFIG['random_seed'])
    torch.cuda.empty_cache()
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)

    # Dataset and Loaders
    full_dataset = SegmentationDataset(CONFIG['image_dir'], CONFIG['mask_dir'], split='train')
    indices = list(range(len(full_dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=CONFIG['test_split'], random_state=CONFIG['random_seed'])
    train_idx, val_idx = train_test_split(train_val_idx, test_size=CONFIG['val_split'], random_state=CONFIG['random_seed'])
    logging.info(f'Train data: {len(train_idx)}, Val data: {len(val_idx)}, Test data: {len(test_idx)}')

    train_dataset = torch.utils.data.Subset(SegmentationDataset(CONFIG['image_dir'], CONFIG['mask_dir'], split='train'), train_idx)
    val_dataset = torch.utils.data.Subset(SegmentationDataset(CONFIG['image_dir'], CONFIG['mask_dir'], split='val'), val_idx)
    test_dataset = torch.utils.data.Subset(SegmentationDataset(CONFIG['image_dir'], CONFIG['mask_dir'], split='test'), test_idx)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_train'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size_val'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_test'], shuffle=False)

    # Models and Training

    
    logging.info(f"Training ...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator, test_metrics = train(generator, discriminator, train_loader, val_loader, test_loader, CONFIG, device, init_type)


    # Print Comparison Table
    logging.info("\nPerformance Comparison: ROIsGAN")
    logging.info("| Model Variant         | Dice (%) | IoU (%) | HD (px) | Precision (%) | Recall (%) | ASSD (px) |")
    logging.info("|-----------------------|----------|---------|---------|---------------|------------|-----------|")
    
    metrics = test_metrics
    logging.info(f"| UNetGAN w/  | {metrics[0]:.2f} ± {metrics[1]:.2f} | {metrics[2]:.2f} ± {metrics[3]:.2f} | "
                    f"{metrics[4]:.1f} ± {metrics[5]:.1f} | {metrics[6]:.2f} ± {metrics[7]:.2f} | "
                    f"{metrics[8]:.2f} ± {metrics[9]:.2f} | {metrics[10]:.1f} ± {metrics[11]:.1f} |")

if __name__ == "__main__":
    main()
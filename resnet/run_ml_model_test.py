import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import os

from utils.custom_dataset_regression_loo import Carbon
from utils.resnet14 import ResNet, BasicBlock
from utils.training_utils_regression import validate

# Set seed.
seed = 7
batch_size = 16
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train_db = Carbon(dataset_foldername, 224, mode='train', subclassname = subclass)
dataset_foldername = 'testing_data'

# remove old data files
try:
    filename = os.path.join(dataset_foldername, 'train_images.csv')
    os.remove(filename)
except OSError:
    pass

try:
    filename = os.path.join(dataset_foldername, 'val_images.csv')
    os.remove(filename)
except OSError:
    pass

val_db = Carbon(dataset_foldername, 224, mode='val', subclassname = 'wifi')

# train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)

model = ResNet(img_channels=6, num_layers=18, block=BasicBlock, num_classes= 1)
ml_best_model_params_path = os.path.join('pretrained_models', 'ml_best_model_params_seed7.pt')
model.load_state_dict(torch.load(ml_best_model_params_path))
model.to(device)

criterion = nn.MSELoss()
fusion_method = 'early'

valid_epoch_loss, valid_epoch_acc, valid_r2, y_preds_v, y_trues_v = validate(
    model, 
    valid_loader, 
    criterion,
    device,
    fusion_method
)

print(f"ML model testing: correlation: {valid_epoch_acc:.3f}, R2: {valid_r2:.3f}, MSE: {valid_epoch_loss:.3f}")
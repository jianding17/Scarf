import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error as mae
# from sklearn.metrics import mean_squared_error as mse

# Training function.
def train(model, trainloader, optimizer, criterion, device, fusion_method, scheduler):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total_num_samples = 0
    y_trues, y_preds = [], []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, wifi_image, labels, metas = data
        meta_oc = metas[:, 1] * 100
        true_labels = labels + meta_oc
        y_trues = y_trues + torch.flatten(true_labels).tolist()
        metas = metas.to(device)
        if fusion_method == 'early':
            combined_input = torch.cat((image, wifi_image), 1)
            combined_input = combined_input.to(device)
        else:
            image = image.to(device)
            wifi_image = wifi_image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        if fusion_method == 'early':
            outputs = model(combined_input, metas)
        else:
            outputs = model(image, wifi_image, metas)
        # Calculate the loss.
        loss = criterion(outputs, labels.reshape(len(labels), 1))
        train_running_loss += loss.item()
        # Calculate the accuracy.
        # _, preds = torch.max(outputs.data, 1)
        preds = outputs.to("cpu")
        true_preds = preds + meta_oc.reshape(len(meta_oc), 1)
        # y_trues = y_trues + torch.flatten(labels).tolist()
        y_preds = y_preds + torch.flatten(true_preds).tolist()
        # y_preds = y_preds + torch.flatten(preds).tolist()
        # train_running_correct += (preds == labels).sum().item()
        total_num_samples += len(labels)
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        # optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    # epoch_acc = 100. * (train_running_correct / total_num_samples)
    corrcoefs = np.corrcoef(y_trues, y_preds)
    corr = corrcoefs[0][1]
    r2 = r2_score(y_trues, y_preds)
    # print(f"Training R2: {r2:.3f}")

    before_lr = optimizer.param_groups[0]["lr"]
    # scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("SGD lr %.6f -> %.6f" % (before_lr, after_lr))
    return epoch_loss, corr, r2, y_preds, y_trues

# Validation function.
def validate(model, testloader, criterion, device, fusion_method):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    total_num_samples = 0
    with torch.no_grad():
        y_trues, y_preds = [], []
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, wifi_image, labels, metas = data
            meta_oc = metas[:, 1] * 100
            metas = metas.to(device)
            if fusion_method == 'early':
                combined_input = torch.cat((image, wifi_image), 1)
                combined_input = combined_input.to(device)          
            else:
                image = image.to(device)
                wifi_image = wifi_image.to(device)
            true_labels = labels + meta_oc
            y_trues = y_trues + torch.flatten(true_labels).tolist()
            # y_trues = y_trues + torch.flatten(labels).tolist()

            labels = labels.to(device)
            # Forward pass.
            if fusion_method == 'early':
                outputs = model(combined_input, metas)
            else:
                outputs = model(image, wifi_image, metas)
            # Calculate the loss.
            loss = criterion(outputs, labels.reshape(len(labels), 1))
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            # _, preds = torch.max(outputs.data, 1)
            preds = outputs.to("cpu")
            true_preds = preds + meta_oc.reshape(len(meta_oc), 1)
            # valid_running_correct += (preds == labels).sum().item()
            total_num_samples += len(labels)
            # preds = preds.to('cpu')
            # y_preds = y_preds + torch.flatten(preds).tolist()
            y_preds = y_preds + torch.flatten(true_preds).tolist()

        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    # epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # epoch_acc = 100. * (valid_running_correct / total_num_samples)
    corrcoefs = np.corrcoef(y_trues, y_preds)
    corr = corrcoefs[0][1]
    r2 = r2_score(y_trues, y_preds)
    abs_diff = [abs(a - b) for a, b in zip(y_trues, y_preds)]
    max_abs_err = max(abs_diff)


    
    # y_mean = np.mean(y_trues)
    # SS_tot = np.sum((np.array(y_trues) - np.array(y_mean))**2)
    # SS_res = np.sum((np.array(y_trues) - np.array(y_preds))**2)
    # r2_2 = 1.0 - (SS_res / SS_tot)
    print(f"Validation R2: {r2:.3f}, maximum abs error: {max_abs_err:.3f}")
    return epoch_loss, corr, r2, y_preds, y_trues

# Test function.
def test(model, testloader, criterion, device):
    model.eval()
    print('Test')
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels.reshape(len(labels), 1))
            test_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
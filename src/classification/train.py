import sys
import shutil
import os
import pickle
import yaml
import importlib
import enum
import copy
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import random

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchio as tio
# import syft as sy
import multiprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(".")
from data_loader import MultiDataLoader 

HOME_PATH = Path.home()

config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)
MODEL_STORAGE = Path('model_states')
VARIABLE_STORAGE = Path('variables')

def initialize_model(device, state_dict=None):
    Model = import_class(config['model']['arch']['function'])
    if config['model']['arch']['args']:
        model_copy = Model(**config['model']['arch']['args'])
    else:
        model_copy = Model()
    if state_dict != None:
        model_copy.load_state_dict(state_dict)
    else:
        model_copy.load_state_dict(model.state_dict())
    model_copy.to(device)
    return model_copy

def initialize_optimizer(model, state_dict=None):
    Opt = import_class(config['hyperparameters']['optimizer'])
    learning_rate = float(config['hyperparameters']['lr'])
    if config['model']['arch']['args']['early_layers_learning_rate']: # if zero then it's freeze
        # Low shallow learning rates instead of freezing
        low_lr_list = []
        high_lr_list = []
        for name,param in model.named_parameters():
            if 'fc' not in name:
                low_lr_list.append(param)
            else:
                high_lr_list.append(param)
        print(f"Initializing optimizer with learning rates {config['model']['arch']['args']['early_layers_learning_rate']} for the early and {learning_rate} for the final layers")
        opt = Opt([
            {'params': low_lr_list, 'lr': float(config['model']['arch']['args']['early_layers_learning_rate'])},
            {'params': high_lr_list, 'lr': learning_rate}
        ], lr=learning_rate)
    else:
        print(f"Layer learning mode set to frozen")
        opt = Opt(model.parameters(), lr=learning_rate)
    if state_dict == None:
        opt.load_state_dict(opt.state_dict())
    else:
        opt.load_state_dict(state_dict)
    return opt

def train(model, dataset, opt, criterion, fold, device, num_epochs=100, fold_splits=[]):
    """The mother of all functions: call this to train your model

    Args:
        model (torch model): your model
        dataset (dict or dataloader class): in centralized case it's a dataloader, in federated it's a dict that assigns centers to data loaders
        opt (torch.opt): optimizer
        criterion (torch.criterion): loss criterion
        fold (int): Assumes crossvalidation, so to keep a log on tensorboard we need this here.
        device (str): 'cpu' or 'cuda'
        num_epochs (int, optional): total number of epochs. Defaults to 100.
        fold_splits (list): a list with the fold index splits (only useful for FL LCO)

    Returns:
        lists with performances per epoch to log
    """

    train_losses, val_losses, confusion_matrices = [], [], []

    fold_model_storage = MODEL_STORAGE.joinpath(f'fold_{fold}')

    # This just doesnt work... Optimizer is problematic if you load, the model is loaded properly
    # if config['model']['continue'] and os.listdir(fold_model_storage):
    #     last_epoch_num = 1
    #     for model_state in os.listdir(fold_model_storage):
    #         if 'epoch' in model_state:
    #             num = int(model_state.split('_')[1])
    #             if num > last_epoch_num:
    #                 last_epoch_num = num
    #     states_dict = torch.load(fold_model_storage.joinpath(f'epoch_{last_epoch_num}'))
    #     start_epoch = states_dict['epoch']+1
    #     model.load_state_dict(states_dict['state_dict'])
    #     # opt.load_state_dict(states_dict['optimizer_state_dict'])
    #     # for state in opt.state.values():
    #     #     for k, v in state.items():
    #     #         if torch.is_tensor(v):
    #     #             state[k] = v.cuda() # .to(device)
    # else:
    start_epoch = 1
    early_stop_counter, best_loss = 0, 10000
    validation_predictions = pd.DataFrame()
    if "epoch_" not in os.listdir(fold_model_storage):
        for epoch in range(start_epoch, num_epochs+1):
            if config['federated']['isTrue']:
                ## Distributed =============================================================================
                if not config['federated']['type'] in ['CIIL', 'SWA']:
                    model.to('cpu')
                if config['cvtype'] == 'LCO':
                    train_indices = fold_splits[0]
                    print(f"Currently training on centers: {train_indices}")
                    trainval_set = {}
                    for train_center in train_indices:
                        trainval_set[train_center] = dataset[train_center]
                else:
                    trainval_set = dataset
                local_models, local_opts, local_weights, predictions, train_loss_avg, val_loss_avg, val_accuracy_avg, confusion_m  = local_train(trainval_set, model, opt, criterion, device, epoch=epoch, fold=fold)

                # average models
                if config['federated']['type'] != 'CIIL':
                    if local_models!=None: # the last epoch is evaluation
                        if not local_weights:
                            local_weights = [1/len(local_models) for _ in local_models] # default case
                        # Model aggregation
                        local_models_states = [lm.state_dict() for lm in local_models] 
                        for i, state in enumerate(local_models_states):
                            if i == 0:
                                global_state = state 
                                print(f'Sum of local weights is : {sum(local_weights)}')
                                for key in global_state:
                                    global_state[key] = local_weights[i]*state[key]
                            else:
                                for key in global_state:
                                    global_state[key] += local_weights[i]*state[key]

                        model.load_state_dict(global_state)

                        # Optimizer aggregation
                        local_opts_states = [lm.state_dict() for lm in local_opts] 
                        for i, state in enumerate(local_opts_states):
                            if i == 0:
                                global_opt_state = state
                                for key in global_opt_state['state']:
                                    global_opt_state['state'][key]['exp_avg'] = local_weights[i]*state['state'][key]['exp_avg']
                                    global_opt_state['state'][key]['exp_avg_sq'] = local_weights[i]*state['state'][key]['exp_avg_sq']
                            else:
                                for key in global_opt_state['state']: # key is step
                                    global_opt_state['state'][key]['exp_avg'] += local_weights[i]*state['state'][key]['exp_avg']
                                    global_opt_state['state'][key]['exp_avg_sq'] += local_weights[i]*state['state'][key]['exp_avg_sq']
                        
                        # global_opt_state = aggregation(local_opts) # aggregating the Adam moments exhibited a 0.5% boost compared to reseting in Spyridon Bakas et al work
                        opt = initialize_optimizer(model, global_opt_state)
                else:
                    model = local_models # it will be just one in this case

                ## Distributed =============================================================================
            else:
                ## Centralized ===========================================================================
                dl = dataset
                training_loader, validation_loader, test_loader = dl.load(fold_index=fold)
                model, opt, _, epoch_losses, _, _ = run_epoch(training_loader, model, opt, criterion, device, epoch=epoch, is_training = True)
                
                train_loss_avg = np.mean(epoch_losses)
                _, _, predictions, epoch_losses, epoch_accuracies, confusion_m = run_epoch(validation_loader, model, opt, criterion, device, epoch=epoch, is_training = False)

                val_loss_avg = np.mean(epoch_losses)
                val_accuracy_avg = np.mean(epoch_accuracies)

                ## Centralized ===========================================================================

            ## Log / Store ======================================================================
            if train_loss_avg != None: # Last epoch for federated is None
                print(f'''
                ========================================================
                Epoch {epoch} finished
                Training loss: {train_loss_avg:0.3f}
                Validation loss: {val_loss_avg:0.3f}, accuracy score:
                {val_accuracy_avg:0.3f}
                ========================================================''')
                
                train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            confusion_matrices.append(confusion_m)
            validation_predictions = validation_predictions.append(predictions)
            ## Log / Store ======================================================================
            if val_loss_avg < best_loss:
                best_epoch = epoch
                best_loss = val_loss_avg
                best_model = initialize_model(device, model.state_dict()) # get a copy of the best model
            elif early_stop_counter == config['hyperparameters']['early_stop_counter']: 
                print("Reached early stop checkpoint")
                break
            else:
                early_stop_counter+=1
                
        torch.save({'epoch': best_epoch,
                    'state_dict': best_model.cpu().state_dict(),
                    # 'optimizer_state_dict': opt.state_dict(),
                        },fold_model_storage.joinpath(f'epoch_{best_epoch}'))
    else:
        files = os.listdir(fold_model_storage)
        for file in files:
            if 'epoch_' in file:
                best_epoch_state = file
                break
        best_epoch = best_epoch_state.split('_')[1]
        print(best_epoch_state)
        checkpoint = torch.load(fold_model_storage.joinpath(f'{best_epoch_state}'))['state_dict']
        model = initialize_model(device, checkpoint)

    # Evaluation on test set:
    print("Initiating testing phase...")
    if config['federated']['isTrue']:
        ## Distributed =============================================================================
        if not config['federated']['type'] in ['CIIL', 'SWA']:
            model.to('cpu')
        if config['cvtype'] == 'LCO':
            test_index = fold_splits[1]
            test_set = {test_index: dataset[test_index]}
        else:
            test_set = dataset
        _, _, _, test_predictions, _, _, test_accuracy_avg, test_confusion_matrix  = local_train(test_set, model, opt, criterion, device, epoch=best_epoch, fold=fold, test=True)
        ## Distributed =============================================================================
    else:
        ## Centralized ===========================================================================
        dl = dataset
        _, _, test_loader = dl.load(fold_index=fold)
        _, _, test_predictions, _, epoch_accuracies, test_confusion_matrix = run_epoch(test_loader, model, opt, criterion, device, epoch=best_epoch, is_training = False)

        test_accuracy_avg = np.mean(epoch_accuracies)

    return train_losses, val_losses, test_predictions, test_confusion_matrix

def local_train(dataset, model, opt, criterion, device, epoch, fold, test=False):
    """This function is used for Federated Learning

    Args:
        dataset (dict): Dataset is a dictionary where the key is the worker / center name and the value is the data corresponding to
        model (torch model): Just put your model
        opt (torch optimizer): Optimizers need to be reset in every epoch
        criterion (torch criterion)
        device (str): 'cpu' or 'cuda'   
        epoch (int): Current epoch
        fold (int): Assumes crossvalidation, so to keep a log on tensorboard we need this here.

    Returns:
        local_models: Store models trained locally to aggregate them later
        The rest of the outputs is just for logging purposes
    """
    train_losses, val_losses, accuracy_scores_v, confusion_m, local_predictions = [], [], [], [], []
    local_models, local_opts, local_weights = [], [], []
    # for name, weight in model.named_parameters():
    #     if weight.requires_grad:
    #         writer_train.add_histogram(f'globalmodel_{name}',weight, epoch)
    SSP_denum = 0
    for i, wd in enumerate(dataset.values()): # iterate over workers and datasets
        worker, dl = wd

        # ========= Redefine the model and optimizer =========
        if config['federated']['type'] in ['CIIL','SWA']: # The model moves from client to client.
            if i == 0:
                model_copy, opt_copy = model, opt
        else: # Federated scheme. We need to redefine the model with the aggregated weights
            Model = import_class(config['model']['arch']['function'])
            if config['model']['arch']['args']:
                model_copy = Model(**config['model']['arch']['args'])
            else:
                model_copy = Model()
            model_copy.load_state_dict(model.state_dict())
            model_copy.to(device)

            Opt = import_class(config['hyperparameters']['optimizer'])
            learning_rate = float(config['hyperparameters']['lr'])
            if config['model']['arch']['args']['early_layers_learning_rate']: # if zero then it's freeze
                # Low shallow learning rates instead of freezing
                low_lr_list = []
                high_lr_list = []
                for name,param in model_copy.named_parameters():
                    if 'fc' not in name:
                        low_lr_list.append(param)
                    else:
                        high_lr_list.append(param)
                print(f"Initializing optimizer with learning rates {config['model']['arch']['args']['early_layers_learning_rate']} for the early and {learning_rate} for the final layers")
                opt_copy = Opt([
                    {'params': low_lr_list, 'lr': float(config['model']['arch']['args']['early_layers_learning_rate'])},
                    {'params': high_lr_list, 'lr': learning_rate}
                ], lr=learning_rate)
            else:
                print(f"Layer learning mode set to frozen")
                opt_copy = Opt(model_copy.parameters(), lr=learning_rate)
            opt_copy.load_state_dict(opt.state_dict())

        # ======================================================Z
        # model_copy = copy.deepcopy(model).to(device) # DOESN'T TRAIN IF YOU USE DEEPCOPY
        # model_copy = type(model)(**config['model']['arch']['args']).to(device) # get a new instance
        # model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
        # model_copy = model.clone()

        print(model_copy.state_dict()['0.fc.bias'])
        # model_copy = model_copy.send(worker)
        if config['cvtype'] == 'LCO':
            training_loader, validation_loader, _ = dl.load(fold_index=fold) # fold index is irrelevant in this case
            if test:
                _, _, test_loader = dl.load(fold_index=fold, LCO_FL_test=True) # this way it will load the entire center as test set
        else:
            training_loader, validation_loader, test_loader = dl.load(fold_index=fold)
        if test:
            _, _, predictions, epoch_losses, epoch_accuracies, epoch_cm = run_epoch(test_loader, model_copy, opt_copy, criterion, device, is_training = False, epoch=epoch, worker=worker)
            #### Log
            with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'rb') as handle:
                log_dict = pickle.load(handle)
            if f'confusion_matrix_{worker}' not in log_dict.keys():
                log_dict[f'confusion_matrix_{worker}'] = {}
            log_dict[f'confusion_matrix_{worker}'][fold] = epoch_cm
            if f'test_predictions_{worker}' not in log_dict.keys():
                log_dict[f'test_predictions_{worker}'] = {}
            log_dict[f'test_predictions_{worker}'][fold] = predictions
            with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'wb') as handle:
                pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            ###
            val_losses.append(epoch_losses)
            accuracy_scores_v.append(epoch_accuracies)
            confusion_m.append(epoch_cm)
            local_predictions.append(predictions)
            continue # skip the rest if test mode

        # Validation comes before training, because we want to use the combined model.
        _, _, predictions, epoch_losses, epoch_accuracies, epoch_cm = run_epoch(validation_loader, model_copy, opt_copy, criterion, device, is_training = False, epoch=epoch, worker=worker)
        
        val_losses.append(epoch_losses)
        accuracy_scores_v.append(epoch_accuracies)
        confusion_m.append(epoch_cm)
        local_predictions.append(predictions)

        if epoch == config['hyperparameters']['num_epochs']:
            continue # skip training if last epoch (it's the final evaluation)

        model_copy, opt_copy, _, epoch_losses, epoch_accuracies, _ = run_epoch(training_loader, model_copy, opt_copy, criterion, device, is_training = True, epoch=epoch, worker=worker)
        
        train_losses.append(epoch_losses) #this is an approximation

        if config['federated']['type'] == 'CIIL': # we just need the final model
            continue

        if not config['federated']['type'] == 'SWA':
            model_copy.to('cpu')
        # model_copy = model_copy.get()
        print(model_copy.state_dict()['0.fc.bias']) # just to check that training proceeds normally, I print the linear layer bias
        local_models.append(model_copy)
        local_opts.append(opt_copy)
        # local_weights are used in different averaging schemes
        if config['federated']['averaging'] == 'SSP':
            SSP_num = len(dl)
            SSP_denum += len(dl)
            local_weights.append(SSP_num)

    local_weights = [SSP_num/SSP_denum for SSP_num in local_weights]
    with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'rb') as handle:
        log_dict = pickle.load(handle)
    if f'val_predictions_{worker}' not in log_dict.keys():
        log_dict[f'val_predictions_{worker}'] = {}
    if fold not in log_dict[f'val_predictions_{worker}'].keys():
        log_dict[f'val_predictions_{worker}'][fold]=[]
    log_dict[f'val_predictions_{worker}'][fold].append(predictions)
    with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'wb') as handle:
        pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    accuracy_scores_v = [item for sublist in accuracy_scores_v for item in sublist] # unfold
    accuracy_scores_v = np.mean(accuracy_scores_v)    
    val_losses = [item for sublist in val_losses for item in sublist] # unfold
    val_losses = np.mean(val_losses)
    confusion_matrix = np.sum(confusion_m, 0)
    all_predictions = [item for sublist in local_predictions for item in sublist] # unfold center predictions

    if config['federated']['type'] == 'CIIL': # we just need the final model
        local_models = model_copy
        
    if epoch == config['hyperparameters']['num_epochs']:
        local_models, local_opts, train_losses = None, None, None
    else:
        train_losses = [item for sublist in train_losses for item in sublist] # unfold
        train_losses = np.mean(train_losses)

    return local_models, local_opts, local_weights, all_predictions, train_losses, val_losses, accuracy_scores_v, confusion_matrix

def run_epoch(loader, model, opt, criterion, device, is_training, epoch, worker=None):
    """run a single epoch

    Args:
        loader (data loader class)
        model (torch model)
        opt (torch optimizer)
        criterion (torch criterion)
        device (str): 'cpu' or 'cuda'
        is_training (bool): this is used to specify training or validation
        epoch (int): current epoch
        worker (syft.VirtualWorker, optional): Not used yet, only needed for real-world federated. Defaults to None.

    Returns:
        model: return model trained for one epoch
        opt: return optimizer, it's necessary since its state also changes
        log of epoch losses / accuracy
    """

    is_federated = config['federated']['isTrue']
    if not is_training:
        model.eval()

    # if is_federated:
    #     model = model.send(worker)
    epoch_losses, epoch_accuracy = [], []
    predictions = pd.DataFrame() 

    for i, batch in enumerate(tqdm(loader)):
        inputs, targets, code = prepare_batch(batch, device)
        if config['model']['arch']['dimensionality'] == '3D':
            inputs = Variable(inputs.float(), requires_grad=True).to(device)
        elif config['model']['arch']['dimensionality'] == '2D':
            slice_index = int(inputs.shape[-1]/2)

            # pick one of the middle 5
            index_list = []
            for num_ in [-2,-1,0,1,2]:
                index_list.append(num_+slice_index)
            chosen_slice = random.choice(index_list)

            inputs = Variable(inputs.float()[...,chosen_slice], requires_grad=True).to(device)
        else:
            raise ValueError(f"Invalid dimensionality for model: {config['model']['arch']['dimensionality']}. Check your config file.")

        targets = Variable(targets).to(device)
        if epoch == 1 and i == 0:
            with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'rb') as handle:
                log_dict = pickle.load(handle)
            log_dict['sample_batch'] = inputs
            with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'wb') as handle:
                pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        # if is_federated:
        #     inputs = inputs.send(worker)
        #     targets = targets.send(worker)
        opt.zero_grad()
        preds = model(inputs)
        if len(preds.shape)>1:
            if preds.shape[1]==1: # binary classification
                preds = preds.squeeze(1)
                targets = targets.float()
        loss = criterion(preds, targets)#((preds - targets)**2).sum()
        if is_training:
            loss.backward()
            opt.step()

        # if is_federated:
        #     loss = loss.get()
        epoch_losses.append(loss.item())
        epoch_accuracy.append(sum(np.array(targets.cpu()) == probabilities_to_labels(preds.cpu()))/len(targets.cpu()))
        if not is_training:
            if i == 0 :
                epoch_cm = confusion_matrix(targets.cpu().detach(), probabilities_to_labels(preds.cpu()), labels=config['data']['labels'])
            else:
                epoch_cm += confusion_matrix(targets.cpu().detach(), probabilities_to_labels(preds.cpu()), labels=config['data']['labels'])
        else:
            epoch_cm = None

        if config['cvtype'] == 'LCO':
            predictions = predictions.append(pd.DataFrame({'predictions': preds.data.cpu(), 'targets': targets.cpu(), 'test_center': config['data']['centres'][fold]}, index=code))
        else:
            predictions = predictions.append(pd.DataFrame({'predictions': preds.data.cpu(), 'targets': targets.cpu(), 'fold': fold}, index=code))
    # if not is_training:
    #     img_grid = torchvision.utils.make_grid(inputs[:,0,...,0].unsqueeze(1), normalize=True, scale_each=True)
    #     writer_train.add_image('MRI scans', img_grid, global_step=epoch)
    # if not is_training:
    #     for i in range(inputs.shape[1]):
    #         img_grid = torchvision.utils.make_grid(inputs[:,i,...,5].unsqueeze(1), normalize=True, scale_each=True)
    #         writer_train.add_image(f'MRI scans channel {i}', img_grid, global_step=epoch)
    # maybe release gradients before exiting function
    return model, opt, predictions, epoch_losses, epoch_accuracy, epoch_cm

def prepare_batch(batch, device):
    """Operates further curation and loads the label encoder to turn strings to encoded format

    Args:
        batch (next(iter(data_loader))): one batch loaded by your data loader
        device (str): 'cpu' or 'cuda'

    Returns:
        inputs, targets
    """
    full_mask = batch['gt']['data']
    if config['model']['arch']['args']['in_ch']==3 and not config['data']['triplicate']:
        if config['data']['concatenateEDES']:
            c12=batch['mri']['data']
            c3=(c12[:,1,...]-c12[:,0,...]).unsqueeze(1)
            inputs = torch.cat((c12,c3), axis=1)
        else:
            c1 = batch['mri']['data']
            lv = torch.zeros(full_mask.shape)
            lv[full_mask==1]=1
            myo = torch.zeros(full_mask.shape)
            myo[full_mask==2]=1
            rv = torch.zeros(full_mask.shape)
            rv[full_mask==3]=1
            if config['data']['only_masks']:
                inputs = torch.cat((lv,myo,rv), axis=1)
            else:
                inputs = torch.cat((c1*lv,c1*myo,c1*rv), axis=1)
    else:
        inputs = batch['mri']['data']
        if config['data']['multiply_by_mask']:
            multiply_mask = torch.zeros(full_mask.shape)
            multiply_mask[full_mask!=0]=1
            inputs = inputs*multiply_mask
        if config['data']['triplicate']:
            inputs = torch.cat((inputs, inputs, inputs), axis=1)

        
    with open(HOME_PATH / Path(config['paths']['misc']).joinpath(config['names']['labelencoder']), 'rb') as handle:
        le = pickle.load(handle)
    with open(HOME_PATH / Path(config['paths']['misc']).joinpath('le.pkl'), 'rb') as handle:
        oldle = pickle.load(handle)
    labels = batch['pathology']
    targets = le.transform(labels)
    targets = torch.as_tensor(targets).long()
    if targets.max() == 1: # Binary Classification needs floats
        targets = targets.float()
    
    # For logging
    code = batch['code']

    return inputs.squeeze(-1), targets, code

def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def probabilities_to_labels(predictions):
    if len(predictions.shape)==1: # if not one hot encoding
        return torch.round(predictions).detach().numpy() #sigmoid outputs
    predictions = predictions.detach().numpy()
    predictions_as_labels = []
    for row in predictions:
        predictions_as_labels.append(np.argmax(row))
    return np.array(predictions_as_labels)



if __name__=='__main__':

    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    log_dict = {} # dictionary for logging
    if config['federated']['isTrue']:
        """FEDERATED: define a dictionary to assign each center to a data loader
        """
        print("Starting training under a federated scheme ...")
        # hook = sy.TorchHook(torch) # remove comment when they fix the bug
        data = {} 
        if config['cvtype'] == 'LCO':
            fold_splits = []
            for centre in config['data']['centres']:
                worker = centre # sy.VirtualWorker(hook, id=centre) # remove comment when they fix the bug
                dl = MultiDataLoader([centre]) # replace with two way split instead
                data[centre]=(worker, dl)
            for i, centre in enumerate(data.keys()): # will be used to index over centers 
                test_indices = centre
                train_indices = [x for j,x in enumerate(data.keys()) if j!=i] # The validation comes out of the centers we train on
                fold_splits.append((train_indices, test_indices))
        else:
            for centre in config['data']['centres']:
                worker = centre # sy.VirtualWorker(hook, id=centre) # remove comment when they fix the bug
                dl = MultiDataLoader([centre])
                fold_splits = dl.get_fold_splits()
                log_dict[f'folds_indices_{centre}'] = fold_splits
                data[centre]=(worker, dl)
    else:
        """CENTRALIZED: initialize a single data loader
        """
        print("Starting centralized training ...")
        data = MultiDataLoader(data_owners=config['data']['centres'])
        fold_splits = data.get_fold_splits()
        log_dict['fold_indices'] = fold_splits

    Criterion = import_class(config['hyperparameters']['criterion'])
    criterion = Criterion()

    # Iterate over all possible folds if crossvalidation is True, else just use 1 fold
    finalfold = 5 if config['data']['crossvalidation'] else 1
    if config['cvtype'] == 'LCO':
        finalfold = len(config['data']['centres'])

    #### Log!
    if VARIABLE_STORAGE.joinpath('log.pkl').exists() and config['model']['continue']:
        print("Continuing from last fold")
        with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'rb') as handle:
            log_dict = pickle.load(handle)
        # if config['test_unseen']:
        #     log_dict['test_predictions'] = pd.DataFrame()
        #     log_dict['final_confusion_matrices'] = []
        #     log_dict['next_fold'] = 0
    else:
        log_dict['next_fold'] = 0
        log_dict['test_predictions'] = pd.DataFrame()
        log_dict['final_train'], log_dict['final_val'], log_dict['final_confusion_matrices'] = [], [], []
        with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    ###

    Model = import_class(config['model']['arch']['function'])
    for fold in range(finalfold):
        # break
        fold_model_storage = MODEL_STORAGE.joinpath(f'fold_{fold}')
        if not fold_model_storage.exists():
            fold_model_storage.mkdir()
        if isinstance(log_dict['next_fold'], str):
            print('Training has finished')
            exit()
        elif log_dict['next_fold']>fold:
            print(f"Fold {fold} has finished, proceeding to fold {fold+1}...")
            continue
        print(f"Fold {fold} iteration initiating...")

        # if len(log_dict['final_accuracy_v']) > fold+1: 
        #     # The value of this key is a list of lists. We put not equal because it will not be a complete list. 
        #     # If it was complete then another list would have been initialized for the next fold
        #     continue

        if config['model']['arch']['args']:
            model = Model(**config['model']['arch']['args'])
        else:
            model = Model()

        if config['model']['pretrained']['isTrue']:
            model.load_state_dict(torch.load(config['model']['pretrained']['weights']), strict=False)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters: {num_parameters}")

        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model.to(device)
        Opt = import_class(config['hyperparameters']['optimizer'])
        # opt = Opt(model.parameters(), lr=config['hyperparameters']['lr'])
        learning_rate = float(config['hyperparameters']['lr'])
        if config['model']['arch']['args']['early_layers_learning_rate']: # if zero then it's freeze
            # Low shallow learning rates instead of freezing
            low_lr_list = []
            high_lr_list = []
            for name,param in model.named_parameters():
                if 'fc' not in name: # Only works for ResNets
                    low_lr_list.append(param)
                else:
                    high_lr_list.append(param)
            print(f"Initializing optimizer with learning rates {config['model']['arch']['args']['early_layers_learning_rate']} for the early and {learning_rate} for the final layers")
            opt = Opt([
                {'params': low_lr_list, 'lr': float(config['model']['arch']['args']['early_layers_learning_rate'])},
                {'params': high_lr_list, 'lr': learning_rate}
            ], lr=learning_rate)
        else:
            print(f"Layer learning mode set to frozen")
            opt = Opt(model.parameters(), lr=learning_rate)

        train_losses, val_losses, test_predictions, confusion_matrices = train(model = model,
            dataset = data,
            opt=opt,
            criterion=criterion,
            device=device,
            num_epochs=config['hyperparameters']['num_epochs'],
            fold=fold,
            fold_splits=fold_splits[fold]) # this one is useful on Federated LCO 

        with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'rb') as handle:
            log_dict = pickle.load(handle)
        #### Log!
        log_dict['final_train'].append(train_losses)
        log_dict['final_val'].append(val_losses)
        log_dict['final_confusion_matrices'].append(confusion_matrices)
        log_dict['test_predictions'] = log_dict['test_predictions'].append(test_predictions)
        log_dict['next_fold'] = fold+1 if fold+1<finalfold else f'finished@fold{fold}_(starting from 0)'
        # if config['cvtype'] == 'LCO':
        #     log_dict['test_center']
        ###

        with open(VARIABLE_STORAGE.joinpath('log.pkl'), 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    # writer_train.close()
    # writer_val.close()

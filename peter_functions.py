import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import random_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from scipy import io
import os

import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_mat_file(num_subject, chop, file_path, option):

    if option == 1:
        file_name = f"{file_path}/Calib_data_{num_subject}_chop_{chop}.mat"
    elif option == 2:
        file_name = f"{file_path}/Eval_data_{num_subject}_chop_{chop}.mat"

    mat_file = io.loadmat(file_name)

    K1 = mat_file['K1']
    A1 = mat_file['A1']

    K2 = mat_file['K2']
    A2 = mat_file['A2']

    Y1 = mat_file['Y1']
    Y2 = mat_file['Y2']

    # K 특성에 대한 Class1 vs Class2 Data 가져오기
    k1 = torch.FloatTensor(K1)
    k1 = k1.transpose(0, 2)

    k2 = torch.FloatTensor(K2)
    k2 = k2.transpose(0, 2)

    # A 특성에 대한 Class1 vs Class2 Data 가져오기
    a1 = torch.FloatTensor(A1)
    a1 = a1.transpose(0, 2)

    a2 = torch.FloatTensor(A2)
    a2 = a2.transpose(0, 2)

    # Y에 대한 Class1 vs Class2 Data 가져오기
    y1 = torch.LongTensor(Y1)
    y2 = torch.LongTensor(Y2)

    k_train = torch.cat([k1, k2], dim=0)
    a_train = torch.cat([a1, a2], dim=0)

    y_train = torch.cat([y1, y2], dim=0)
    y_train = y_train - 1  # y를 0~1의 정수로 만들어야함.

    return k_train, a_train, y_train


def build_dataset(batch_size, k_train, a_train, y_train, k_test, a_test,
                  y_test):
    dataset_train = TensorDataset(k_train, a_train,
                                  y_train)  # 각 tensor의 첫번째 dim이 일치해야한다
    dataset_test = TensorDataset(k_test, a_test,
                                 y_test)  # 각 tensor의 첫번째 dim이 일치해야한다

    # Data Split
    dataset_size = len(dataset_train)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(dataset_train,
                                                [train_size, valid_size])

    train_DL = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
    valid_DL = DataLoader(valid_dataset, batch_size=valid_size, shuffle=False)

    test_DL = DataLoader(dataset_test, batch_size=len(dataset_test))

    return train_DL, valid_DL, test_DL


def build_dataset_tf(batch_size, k_train, a_train, y_train, k_test, a_test,
                     y_test):
    dataset_train = TensorDataset(k_train, a_train,
                                  y_train)  # 각 tensor의 첫번째 dim이 일치해야한다
    dataset_test = TensorDataset(k_test, a_test,
                                 y_test)  # 각 tensor의 첫번째 dim이 일치해야한다

    # Data Split
    dataset_size = len(dataset_train)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size

    dataset_size2 = len(dataset_test)
    test_size = int(dataset_size2 * 0.8)
    tf_size = dataset_size2 - test_size

    train_dataset, valid_dataset = random_split(dataset_train,
                                                [train_size, valid_size])
    test_dataset, tf_dataset = random_split(dataset_test, [test_size, tf_size])

    train_DL = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
    valid_DL = DataLoader(valid_dataset, batch_size=valid_size, shuffle=False)

    test_DL = DataLoader(test_dataset, batch_size=len(test_dataset))
    tf_DL = DataLoader(tf_dataset, batch_size=len(tf_dataset))

    return train_DL, valid_DL, test_DL, tf_DL


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate,
                              momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer


def Train(model, train_DL, val_DL, config=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95**epoch,
        last_epoch=-1,
        verbose=False)

    for epoch in range(config.epochs):
        rloss = 0
        model.train()
        for _, samples in enumerate(train_DL):

            k_train_mb, a_train_mb, y_train_mb = samples

            hidden_k = torch.zeros(1,
                                   config.batch_size,
                                   config.hidden_size,
                                   requires_grad=True).to(DEVICE)
            cell_k = torch.zeros(1,
                                 config.batch_size,
                                 config.hidden_size,
                                 requires_grad=True).to(DEVICE)
            hidden_a = torch.zeros(1,
                                   config.batch_size,
                                   config.hidden_size,
                                   requires_grad=True).to(DEVICE)
            cell_a = torch.zeros(1,
                                 config.batch_size,
                                 config.hidden_size,
                                 requires_grad=True).to(DEVICE)

            # Forward
            output = model((hidden_k, cell_k), (hidden_a, cell_a),
                           (k_train_mb.to(DEVICE), a_train_mb.to(DEVICE)))

            # Cost
            loss = criterion(output.to(DEVICE),
                             y_train_mb.squeeze().to(DEVICE))

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_b = loss.item() * config.batch_size
            rloss += float(loss_b)
        model.eval()
        with torch.no_grad():
            # epoch loss
            loss_e = rloss / len(train_DL.dataset)

            # Validation
            k_valid, a_valid, y_valid = next(iter(val_DL))

            hidden_k = torch.zeros(1, len(val_DL.dataset),
                                   config.hidden_size).to(DEVICE)
            cell_k = torch.zeros(1, len(val_DL.dataset),
                                 config.hidden_size).to(DEVICE)
            hidden_a = torch.zeros(1, len(val_DL.dataset),
                                   config.hidden_size).to(DEVICE)
            cell_a = torch.zeros(1, len(val_DL.dataset),
                                 config.hidden_size).to(DEVICE)

            output = model((hidden_k, cell_k), (hidden_a, cell_a),
                           (k_valid.to(DEVICE), a_valid.to(DEVICE)))
            prediction = output.argmax(dim=1)
            correct = prediction.eq(y_valid.view_as(prediction)).sum().item()

            # Wandb log
            wandb.log({"Loss": loss_e})
            wandb.log({"Val accuracy": correct / len(val_DL.dataset)})

            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch}, train loss: {round(loss_e,3)}, Val accuracy: {round(correct/len(val_DL.dataset),3)}"
                )
        scheduler.step()


def Train_tf(model, train_DL, val_DL=None, config=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95**epoch,
        last_epoch=-1,
        verbose=False)

    for epoch in range(config.epochs / 2):
        rloss = 0
        model.train()
        for _, samples in enumerate(train_DL):

            k_train_mb, a_train_mb, y_train_mb = samples

            hidden_k = torch.zeros(1,
                                   config.batch_size,
                                   config.hidden_size,
                                   requires_grad=True).to(DEVICE)
            cell_k = torch.zeros(1,
                                 config.batch_size,
                                 config.hidden_size,
                                 requires_grad=True).to(DEVICE)
            hidden_a = torch.zeros(1,
                                   config.batch_size,
                                   config.hidden_size,
                                   requires_grad=True).to(DEVICE)
            cell_a = torch.zeros(1,
                                 config.batch_size,
                                 config.hidden_size,
                                 requires_grad=True).to(DEVICE)

            # Forward
            output = model((hidden_k, cell_k), (hidden_a, cell_a),
                           (k_train_mb.to(DEVICE), a_train_mb.to(DEVICE)))

            # Cost
            loss = criterion(output.to(DEVICE),
                             y_train_mb.squeeze().to(DEVICE))

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_b = loss.item() * config.batch_size
            rloss += float(loss_b)

        model.eval()
        with torch.no_grad():
            # epoch loss
            loss_e = rloss / len(train_DL.dataset)

            if val_DL is not None:
                # Validation
                k_valid, a_valid, y_valid = next(iter(val_DL))

                hidden_k = torch.zeros(1, len(val_DL.dataset),
                                       config.hidden_size).to(DEVICE)
                cell_k = torch.zeros(1, len(val_DL.dataset),
                                     config.hidden_size).to(DEVICE)
                hidden_a = torch.zeros(1, len(val_DL.dataset),
                                       config.hidden_size).to(DEVICE)
                cell_a = torch.zeros(1, len(val_DL.dataset),
                                     config.hidden_size).to(DEVICE)

                output = model((hidden_k, cell_k), (hidden_a, cell_a),
                               (k_valid.to(DEVICE), a_valid.to(DEVICE)))
                prediction = output.argmax(dim=1)
                correct = prediction.eq(
                    y_valid.view_as(prediction)).sum().item()
                wandb.log({"Val accuracy": correct / len(val_DL.dataset)})

            # Wandb log
            wandb.log({"Loss_tf": loss_e})

            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch}, train loss: {round(loss_e,3)}, Val accuracy: {round(correct/len(val_DL.dataset),3)}"
                )
        scheduler.step()


def Test(model, test_DL, config=None):
    model.eval()
    with torch.no_grad():
        for _, samples in enumerate(test_DL):
            k_train_mb, a_train_mb, y_train_mb = samples

            hidden_k = torch.zeros(1, len(test_DL.dataset),
                                   config.hidden_size).to(DEVICE)
            cell_k = torch.zeros(1, len(test_DL.dataset),
                                 config.hidden_size).to(DEVICE)
            hidden_a = torch.zeros(1, len(test_DL.dataset),
                                   config.hidden_size).to(DEVICE)
            cell_a = torch.zeros(1, len(test_DL.dataset),
                                 config.hidden_size).to(DEVICE)

            output = model((hidden_k, cell_k), (hidden_a, cell_a),
                           (k_train_mb.to(DEVICE), a_train_mb.to(DEVICE)))
            prediction = output.argmax(dim=1)
            correct = prediction.eq(
                y_train_mb.view_as(prediction)).sum().item()
            print(
                f"Evaluation accuracy: {round(correct/len(test_DL.dataset),3)}"
            )
            wandb.log({"Test accuracy": correct / len(test_DL.dataset)})

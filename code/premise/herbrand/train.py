from parsing.argparser import parse_train_arguments, fill_default_values
from os.path import isfile, join
import pickle, os, random
from premise.herbrand.PremiseDataset import PremiseDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from premise.herbrand.MLP import MLP
from datetime import date
from sklearn.metrics import classification_report, confusion_matrix

def binary_acc(y_pred, y_test):
    # print(y_pred.shape)
    # print(y_test.shape)
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

if __name__ == '__main__':
    parser = parse_train_arguments()
    parser.add_argument("--dataset_folder", type=str,
                        default='../../../data/mizar26K_dataset_for_premise_selection_cnf_cashed/CachedParsesByExternalParser')
    parser.add_argument("--model_folder", type=str, default='../models/')
    parser.add_argument("--num_of_problems_to_train", type=int, default='100')
    parser.add_argument("--model_name_prefix", type=str, default='')

    parsed_args = parser.parse_args()
    # args = fill_default_values(parsed_args)
    parsed_args.append_age_features = 0

    dataset_folder = parsed_args.dataset_folder
    model_folder = parsed_args.model_folder
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    num_of_problems_to_train = parsed_args.num_of_problems_to_train
    model_name_prefix = parsed_args.model_name_prefix

    files = os.listdir(dataset_folder)[:num_of_problems_to_train]
    random.shuffle(files)
    num_train = int(len(files)*0.8)
    train_files = files[:num_train]
    valid_files = files[num_train:]

    train_dataset = PremiseDataset(dataset_folder, train_files, parsed_args)
    valid_dataset = PremiseDataset(dataset_folder, valid_files, parsed_args)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=10, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=10, shuffle=True)

    print(f'Number of train files: {len(train_files)}, # samples in loader {len(train_loader)}')
    print(f'Number of test files: {valid_files}, # samples in loader {len(valid_loader)}')


    # append_age_features = 4
    input_size = (parsed_args.herbrand_vector_size)  * 2
    num_classes = 2

    #logistic regression model
    # model = nn.Linear(input_size, num_classes)
    # criterion = nn.CrossEntropyLoss()

    # #MLP
    model = MLP(input_size, hidden_size=input_size * 2, output_size=1)
    criterion = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss has sigmoid already?

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    total_files = len(train_loader)
    model.train()
    num_ex_train = 0
    for epoch in range(num_epochs):
        ep_loss = 0
        ep_acc = 0
        for i, (conj_premises, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                conj_premises = conj_premises.to(device)
                labels = labels.to(device)

            # print(labels)
            outputs = model(conj_premises.float())
            # print(outputs.shape) #torch.Size([B, 1])
            # print(labels.shape) #torch.Size([B])
            # print(outputs.squeeze().shape) #torch.Size([B])
            if type(model) == MLP:
                loss = criterion(outputs.squeeze(), labels.float())
            else:
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            accuracy = binary_acc(outputs.squeeze().float(), labels.float())
            ep_acc += accuracy.item()
            num_ex_train += labels.shape[0]
        print(f'Epoch {epoch + 0:03}: | Loss: {ep_loss / len(train_loader):.5f} | Acc: {ep_acc / len(train_loader):.3f}')

    #TEsting
    model.eval()
    y_pred_list = []
    y_all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for (conj_premises, labels) in valid_loader:
            if torch.cuda.is_available():
                conj_premises = conj_premises.to(device)
                labels = labels.to(device)#torch.Size([B])
            outputs = model(conj_premises.float())#torch.Size([B, 1])
            outputs = torch.sigmoid(outputs)
            y_pred = torch.round(outputs)


            y_pred_ = y_pred.squeeze().cpu().int()#.numpy().astype(int)
            labels_ = labels.cpu().int()#.numpy().astype(int)
            correct += (y_pred_ == labels_).sum()
            print(y_pred_)
            print(type(y_pred_))
            # print(labels_)
            # print(correct)
            # print(labels.cpu().numpy())
            # print(y_pred_.cpu().numpy())
            # for i in enumerate(labels_):
            #     if y_pred_[i] == labels_[i]:
            #         correct += 1
            total += labels.size(0)

            y_pred_list.extend(y_pred.cpu().numpy())
            y_all_labels.extend(labels.float().cpu().numpy())
            # # print('outputs: ', outputs)
            # _, predicted = torch.max(outputs.data, 1)
            # # print('Predicted: ', predicted)
            # # print('Labels: ', labels)
        # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # y_all_labels = [a.squeeze().tolist() for a in y_all_labels]
        # print(y_pred_list)
        # print(y_all_labels)

    print('Accuracy of the model on the {} test samples: {}%'.format(len(valid_loader), 100 * correct / total))
    print(classification_report(y_all_labels, y_pred_list))
    print(confusion_matrix(y_all_labels, y_pred_list))
    print(f'Number of train files: {len(train_files)}, # samples in loader {len(train_loader)}, {num_ex_train}')
    print(f'Number of test files: {valid_files}, # samples in loader {len(valid_loader)}, {len(y_all_labels)}')

    d3 = date.today().strftime("%m_%d_%y")
    torch.save(model.state_dict(), model_folder + '/'+model_name_prefix + '_' + d3 +'_model.ckpt')
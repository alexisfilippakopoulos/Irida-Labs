import socket
import threading
import pickle
import sys
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
import time
from fl_plan import FL_Plan
from client_models import ClientModel, ClientClassifier
from custom_dataset import CustomDataset
import pandas as pd


# Events to ensure synchronization
start_label_event = threading.Event()
labels_recvd_event = threading.Event()
fl_plan_event = threading.Event()
start_training_event = threading.Event()
aggr_recvd_event = threading.Event()

#Constants
VALIDATION_SPLIT = 0.2

def tictoc(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        func_result = func(self, *args, **kwargs)
        end = time.time()
        return func_result, end - start
    return wrapper

class Client:
    def __init__(self, server_ip, server_port, client_ip, client_port):
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.client_ip = client_ip
        self.client_port = int(client_port)
        self.client_model = ClientModel()
        self.classifier_model = ClientClassifier()
        self.server_labels = []
        self.true_labs = []
        self.event_dict = {'LABELS': labels_recvd_event, 'LABEL_EVENT': start_label_event, 'PLAN': fl_plan_event, 'TRAIN': start_training_event, 'AGGR_MODELS': aggr_recvd_event}
        self.device = self.get_device()
        self.epoch_stats_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_acc', 'train_time', 'val_time'])
        print(f'Using {self.device}')

    def create_socket(self):
        """
        Binds the client-side socket to enable communication and connects with the server-side socket
        to establish communication.
        """
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.client_ip, self.client_port))
            self.server_socket.connect((self.server_ip, self.server_port))
            print(f'[+] Connected successfully with server at ({self.server_ip}, {self.server_port})')
        except socket.error as error:
            print(f'Socket initialization failed with error:\n{error}')
            print(self.server_socket.close())

    def listen_for_messages(self):
        """
        Communication thread. Listens for incoming messages from the server.
        """
        data_packet = b''
        try:
            while True:
                data_chunk = self.server_socket.recv(4096)
                if not data_chunk:
                    break
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                    self.handle_packets(data_packet)
                    data_packet = b''
        except socket.error as error:
            print(f'Error receiving data:\n{error}')

    def send_packet(self, data):
        """
        Packs and sends a payload of data to the server.
        Args:
            data: payload of data to be sent.
        """
        try:
            self.server_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')

    def handle_packets(self, data_packet: bytes):
        """
        Handle each complete data packet that arrives and set the corresponding event 
        """
        data = data_packet.split(b'<START>')[1].split(b'<END>')[0]
        data = pickle.loads(data)
        header = list(data.keys())[0]
        if header == 'LABELS':
            self.server_labels.append(data[header])
        elif header == 'PLAN':
            self.fl_plan = data[header]
            self.handle_fl_plan()
            self.send_packet(data={'OK': b''})
        elif header == 'AGGR_MODELS':
            self.client_model.load_state_dict(data[header][0])
            self.classifier_model.load_state_dict(data[header][1])
            print("[+] Received and loaded aggregated weights")
        self.event_dict[header].set()

    def handle_fl_plan(self):
        """
        Handle the FL plan.
        Extract all information and set the appropriate client parameters.
        """
        str_args = {'optimizer': {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}, 'criterion': {'crossentropy': nn.CrossEntropyLoss}}
        try:
            if self.fl_plan.CRITERION in str_args['criterion']:
                self.criterion = str_args['criterion'][self.fl_plan.CRITERION]()
            else:
                raise ValueError(f'Unsupported Loss Function: {self.fl_plan.CRITERION}')
            if self.fl_plan.OPTIMIZER in str_args['optimizer']:
                self.model_optimizer = str_args['optimizer'][self.fl_plan.OPTIMIZER](params=self.client_model.parameters(), lr=client.fl_plan.LEARNING_RATE)
                self.classifier_optimizer = str_args['optimizer'][self.fl_plan.OPTIMIZER](params=self.classifier_model.parameters(), lr=client.fl_plan.LEARNING_RATE)
            else:
                raise ValueError(f'Unsupported Optimizer: {self.fl_plan.OPTIMIZER}')
            self.client_model.load_state_dict(self.fl_plan.model_weights)
            self.classifier_model.load_state_dict(self.fl_plan.classifier_weights)
            print('[+] Loaded FL plan successfully')
            print(self.fl_plan)
        except ValueError as e:
            print(e)
        

    def get_dataset(self, shard_id: int):
        """
        Downloads FashionMNIST dataset.
        Shards the data according to the shard_id argument and allocates the specified subset of data.
        Returns:
            train_subset: Training dataset
            test_subset: Testing dataset
        """
        shards = {'0': [0, 1, 5, 6, 9], '1': [2, 3, 4, 7, 8], '2': [0, 1, 5, 7, 8], '3': [2, 3, 4, 6, 9]}
        #shards = {'0': [0, 1], '1': [2, 3], '2': [4, 5], '3': [6, 7], '4': [8, 9]}
        training_indices = []
        testing_indices = []

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ), )])
        training_data = datasets.FashionMNIST(download=True, root='Implementation/data/', train=True, transform=transform)
        testing_data = datasets.FashionMNIST(download=True, root='Implementation/data/', train=False, transform=transform)
        
        for i in range(len(training_data.targets)):
            if training_data.targets[i] in shards[shard_id]:
                training_indices.append(i)

        for i in range(len(testing_data.targets)):
            if testing_data.targets[i] in shards[shard_id]:
                testing_indices.append(i)

        train_subset = Subset(dataset=training_data, indices=training_indices)
        test_subset = Subset(dataset=testing_data, indices=testing_indices)
        return train_subset, test_subset
    
    def get_dataloader(self, data: datasets, batch_size: int, shuffle: bool, split_flag: bool = False):
        """
        Creates a DataLoader object for a specified dataset.
        Args:
            data: Dataset to be used.
        Returns:
            DataLoader for the specidied dataset.
        """
        if split_flag:
            training_data, validation_data = random_split(data, [int((1 - VALIDATION_SPLIT) * len(data)), int(VALIDATION_SPLIT * len(data))])
            return DataLoader(dataset=training_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=shuffle), DataLoader(dataset=validation_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=shuffle)
        
        return DataLoader(dataset=data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=shuffle)

    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_labels(self, train_dl: DataLoader):
        print('[+] Waiting for labeling process with server')
        start_label_event.wait()
        start_label_event.clear()
        print('[+] Starting labeling process with server')
        with torch.inference_mode():
            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.true_labs.append(labels)
                split_features_maps = self.client_model(inputs)
                #if i == len(train_dl) - 1:
                if i == 3:
                    self.send_packet(data={'final_model_outputs': [split_features_maps, len(train_dl)]})
                    labels_recvd_event.wait()
                    break
                else:
                    self.send_packet(data={'model_outputs': split_features_maps})
                labels_recvd_event.wait()
                labels_recvd_event.clear()
        print('[+] Finished labeling')
        self.server_labels = torch.cat(tensors=self.server_labels, dim=0)
        self.true_labs = torch.cat(tensors=self.true_labs, dim=0)

    @tictoc
    def train_one_epoch(self, train_dl: DataLoader):
        self.client_model.train(), self.classifier_model.train()
        curr_loss = 0.
        for i, (inputs, labels) in enumerate(train_dl):
            self.model_optimizer.zero_grad(), self.classifier_optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.classifier_model(self.client_model(inputs))
            loss = self.criterion(outputs, labels)
            loss.backward()
            curr_loss += loss.item()
            self.model_optimizer.step(), self.classifier_optimizer.step()
        print(f'\t[+] Average Training Loss: {(curr_loss / len(train_dl)): .2f}')
        return curr_loss / len(train_dl)

    @tictoc
    def validate(self, val_dl: DataLoader):
        self.client_model.eval(), self.classifier_model.eval()
        curr_vloss = 0.
        corr = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_dl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.classifier_model(self.client_model(inputs))
            vloss = self.criterion(outputs, labels)
            curr_vloss += vloss.item()
            _, preds = torch.max(outputs.detach(), dim=1)
            corr += (preds == labels).sum().item()
            total += labels.size(0)
        del inputs, labels, outputs
        avg_vloss = curr_vloss / len(val_dl)
        val_acc = corr / total
        print(f'\t[+] Average Validation Loss: {avg_vloss: .2f}\n\t[+] Average Validation Accuracy: {val_acc: .2%}')
        return avg_vloss, val_acc
    

if __name__ == '__main__':
# To execute, server_ip, server_port and client_ip, client_port must be specified from the cl.
    if len(sys.argv) != 6:
        print('Incorrect number of command-line arguments\nTo execute, server_ip, server_port, client_ip, client_port and shard_id must be specified from the cl.')
        sys.exit(1)

    client = Client(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    client.create_socket()
    threading.Thread(target=client.listen_for_messages).start()
    training_data, testing_data = client.get_dataset(shard_id=sys.argv[5])
    fl_plan_event.wait()
    fl_plan_event.clear()
    train_dl, val_dl = client.get_dataloader(data=training_data, batch_size=client.fl_plan.BATCH_SIZE, shuffle=False, split_flag=True)
    test_dl = client.get_dataloader(data=testing_data, batch_size=client.fl_plan.BATCH_SIZE, shuffle=False)
    client.get_labels(train_dl=train_dl)
    #print('Time to label: ', end - start)
    #print(len(train_dl))
    #print(len(training_data))
    #print(len(train_dl) * client.fl_plan.BATCH_SIZE)
    # Custom dataset with server preds
    """training_data = CustomDataset(training_data, client.server_labels)
    train_dl = client.get_dataloader(data=training_data, batch_size=BATCH_SIZE, shuffle=True)
    print('[+] Custom dataset successfully created')
    print(len(training_data))
    print(len(train_dl))"""
    print('[+] Waiting for training signal')
    start_training_event.wait()
    start_training_event.clear()
    for e in range(client.fl_plan.GLOBAL_TRAINING_ROUNDS):
        print(f'[+] Started training for global epoch: {e}')
        avg_train_loss, train_time = client.train_one_epoch(train_dl=train_dl)
        (avg_vloss, val_acc), val_time = client.validate(val_dl=val_dl)
        client.send_packet(data={'UPDATED_WEIGHTS': [client.client_model.state_dict(), client.classifier_model.state_dict()]})
        client.epoch_stats_df.loc[len(client.epoch_stats_df)] = {'epoch': e + 1, 'train_loss': avg_train_loss, 'val_loss': avg_vloss, 'val_acc': val_acc, 'train_time': train_time, 'val_time': val_time}
        client.epoch_stats_df.to_csv(path_or_buf=f'client_{client.client_port}.csv')
        print(f'[+] Waiting for aggregated global model')
        aggr_recvd_event.wait()
        aggr_recvd_event.clear()




    


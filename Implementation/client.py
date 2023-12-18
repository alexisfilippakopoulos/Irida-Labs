import socket
import threading
import pickle
import sys
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# Events to ensure synchronization
start_label_event = threading.Event()
labels_recvd_event = threading.Event()

#Constants
BATCH_SIZE = 64

class Client:
    def __init__(self, server_ip, server_port, client_ip, client_port):
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.client_ip = client_ip
        self.client_port = int(client_port)
        self.client_model = ClientModel()
        self.classifier_model = ClientClassifier()
        self.server_labels = []
        self.event_dict = {'LABELS': labels_recvd_event, 'LABEL_EVENT': start_label_event}
        self.device = self.get_device()
        self.criterion = nn.CrossEntropyLoss()
        self.model_optimizer = torch.optim.SGD(params=self.client_model.parameters(), lr=0.01)
        self.classifier_optimizer = torch.optim.SGD(params=self.classifier_model.parameters(), lr=0.01)
        self.data_features = []
        print(f'Using {self.device}')

    def create_socket(self):
        """
        Binds the client-side socket to enable communication and connects with the server-side socket
        to establish communication.
        """
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.server_ip, self.client_port))
            self.server_socket.connect((self.client_ip, self.server_port))
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
        data = data_packet.split(b'<START>')[1].split(b'<END>')[0]
        data = pickle.loads(data)
        header = list(data.keys())[0]
        if header == 'LABELS':
            self.server_labels.append(data['LABELS'])
            self.event_dict['LABELS'].set()
        elif header == 'LABEL_EVENT':
            start_label_event.set()
            print('[+] Started training with server')
            

    def get_dataset(self):
        """
        Downloads CIFAR10 dataset.
        Returns:
            training_data: Training dataset
            testing_data: Testing dataset
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        training_data = datasets.CIFAR10(download=True, root='/data', train=True, transform=transform)
        testing_data = datasets.CIFAR10(download=True, root='/data', train=False, transform=transform)
        return training_data, testing_data
    
    def get_dataloader(self, data: datasets, batch_size: int):
        """
        Creates a DataLoader object for a specified dataset.
        Args:
            data: Dataset to be used.
        Returns:
            DataLoader for the specidied dataset.
        """
        return DataLoader(dataset=data, batch_size=batch_size, num_workers=2, pin_memory=True)

    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_labels(self, train_dl: DataLoader):
        start_label_event.wait()
        start_label_event.clear()
        with torch.inference_mode():
            for i, (inputs, labels) in enumerate(train_dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                split_features_maps = self.client_model(inputs)
                self.send_packet(data={'model_outputs': split_features_maps})
                labels_recvd_event.wait()
                labels_recvd_event.clear()
        self.send_packet(data={'FINISH': b''})
        print('[+] Gathered labels')
        print(len(torch.cat(self.server_labels, 0)))

    def train_one_epoch(self, train_dl):
        self.client_model.train(), self.classifier_model.train()
        for i, (inputs, labels) in enumerate(train_dl):
            self.model_optimizer.zero_grad(), self.classifier_optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.classifier_model(self.client_model(inputs))
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.model_optimizer.step(), self.classifier_optimizer.step()
               

class ClientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5*5*256, out_features=10)
        
    def forward(self, x):
        x = self.fc1(torch.flatten(x, 1))
        return x

class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        return x
    

if __name__ == '__main__':
# To execute, server_ip, server_port and client_ip, client_port must be specified from the cl.
    if len(sys.argv) != 5:
        print('Incorrect number of command-line arguments\nTo execute, server_ip, server_port, client_ip and client_port must be specified from the cl.')
        sys.exit(1)

    client = Client(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    client.create_socket()
    threading.Thread(target=client.listen_for_messages).start()
    training_data, testing_data = client.get_dataset()
    train_dl, test_dl = client.get_dataloader(data=training_data, batch_size=BATCH_SIZE), client.get_dataloader(data=testing_data, batch_size=BATCH_SIZE)
    #client.train(train_dl=train_dl)
    print('Data: ', len(training_data))
    print('DL: ', len(train_dl))
    client.get_labels(train_dl=train_dl)
    print(len(client.server_labels))


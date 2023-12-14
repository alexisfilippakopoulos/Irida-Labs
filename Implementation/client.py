import socket
import threading
import pickle
import sys
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Client:
    def __init__(self, server_ip, server_port, client_ip, client_port):
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.client_ip = client_ip
        self.client_port = int(client_port)

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

    def listen_for_messages(self):
        """
        Communication thread. Listens for incoming messages from the server.
        """
        data_packet = b''
        while True:
            try:
                data_chunk = self.server_socket.recv(4096)
                if not data_chunk:
                    break
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                    data = data_packet.split(b'<START>')[1].split(b'<END>')[0]
                    print(pickle.loads(data))
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

    def extract_payload(self, data_packet: bytes):
        pass

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
        return DataLoader(dataset=data, batch_size=batch_size, num_workers=3, pin_memory=True)

if __name__ == '__main__':
# To execute, server_ip, server_port and client_ip, client_port must be specified from the cl.
    if len(sys.argv) != 5:
        print('Incorrect number of command-line arguments\nTo execute, server_ip, server_port and client_ip, client_port must be specified from the cl.')
        sys.exit(1)

    client = Client(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    client.create_socket()
    threading.Thread(target=client.listen_for_messages).start()
    client.send_packet(data='hey')
    training_data, testing_data = client.get_dataset()


import socket
import threading
import pickle
import sys
import torch.nn as nn
import sqlite3
import torch

# Events to ensure synchronization
features_recvd_event = threading.Event()
sufficient_clients_event = threading.Event()
epoch_finished = threading.Event()

# Constants
MIN_PARTICIPANTS = 1

class Server:
    def __init__(self, server_ip, server_port):
        self.connected_clients = {}
        self.trained_clients = []
        self.ip = server_ip
        self.port = int(server_port)
        self.server_db = 'Implementation/server_data/server_db.db'
        self.server_model = ServerModel()
        self.event_dict = {'model_outputs': features_recvd_event, 'enough_clients': sufficient_clients_event, 'FINISH': epoch_finished}
        self.device = self.get_device()
        print(f'Using {self.device}')
        #self.client_model = ClientModel()
        #self.classifier_model = ClientClassifier()

    def create_db_schema(self):
        """
        Creates the server-side database schema.
        """
        clients_table = """
        CREATE TABLE clients(
        id INT PRIMARY KEY,
        ip VARCHAR(50),
        port INT    
        )
        """
        training_table = """
        CREATE TABLE training(
        client_id INT PRIMARY KEY,
        model_outputs BLOB,
        updated_weights BLOB,
        FOREIGN KEY (client_id) REFERENCES clients (id)
        )
        """
        self.execute_query(query=clients_table) if self.check_table_existence(clients_table) else None
        self.execute_query(query=training_table) if self.check_table_existence(training_table) else None
        print('[+] Database schema created/loaded successsfully')

    def check_table_existence(self, target_table: str):
        """
        Checks if a specific table exists within the database.
        Args:
            target_table: Table to look for.
        Returns:
            True or False depending on existense.
        """
        query = "SELECT name FROM sqlite_master WHERE type ='table'"
        tables = self.execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        exists = any(table[0] == target_table for table in tables)
        return exists
    
    def execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes a given query. Either for retreival or update purposes.
        Args:
            query: Query to be executed
            values: Query values
            fetch_data_flag: Flag that signals a retrieval query
            fetch_all_flag: Flag that signals retrieval of all table data or just the first row.
        Returns:
            The data fetched for a specified query. If it is not a retrieval query then None is returned. 
        """
        try:
            connection = sqlite3.Connection(self.server_db)
            cursor = connection.cursor()
            cursor.execute(query, values) if values is not None else cursor.execute(query)
            fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
            connection.commit()
            connection.close()        
            return fetched_data
        except sqlite3.Error as error:
            print(f'Query execution failed with error:\n{error}')

    def create_socket(self):
        """
        Binds the server-side socket to enable communication.
        """
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.ip, self.port))
            print(f'[+] Server initialized successfully at {self.ip, self.port}')
        except socket.error as error:
            print(f'Socket initialization failed with error:\n{error}')

    def listen_for_connections(self):
        """
        Listening to the server-side port for incoming connections from clients.
        Creates a unique communication thread for each connected client. 
        """
        try:
            self.server_socket.listen()
            while True:
                client_socket, client_address = self.server_socket.accept()
                client_id = self.handle_connections(client_address, client_socket)
                threading.Thread(target=server.listen_for_messages, args=(client_socket, client_id)).start()
        except socket.error as error:
            print(f'Connection handling thread failed:\n{error}')
            
    def listen_for_messages(self, client_socket: socket.socket, client_id: int):
        """
        Client-specific communication thread. Listens for incoming messages from a unique client.
        Args:
            client_socket: socket used from a particular client to establish communication.
        """
        data_packet = b''
        try:
            while True:
                data_chunk = client_socket.recv(4096)
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        #data = data_packet.split(b'<START>')[1].split(b'<END>')[0]
                        threading.Thread(target=self.store_data, args=(data_packet, client_id)).start()
                        data_packet = b'' 
                if not data_chunk:
                    break
        except socket.error as error:
            print(f'Error receiving data from {client_socket}:\n{error}')
            client_socket.close()
    
    def handle_connections(self, client_address: tuple, client_socket: socket.socket):
        """
        When a client connects -> Add him on db if nonexistent and append to connected_clients list
        Args: Tuple (client_ip, client_port)
        """
        client_ip, client_port = client_address
        query = """
        SELECT id
        FROM clients
        WHERE ip = ? AND port = ?
        """
        exists = self.execute_query(query=query, values=(client_ip, client_port), fetch_data_flag=True, fetch_all_flag=True)
        if len(exists) == 0:
            query = """
            SELECT id FROM clients ORDER BY id DESC LIMIT 1;
            """
            last_id = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
            client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
            query = """
            INSERT INTO clients (id, ip, port) VALUES (?, ?, ?)
            """
            self.execute_query(query=query, values=(client_id, client_ip, client_port))
        else:
            client_id = exists[0][0]
        self.connected_clients[client_id] = (client_address, client_socket)
        print(f'[+] Client {client_address} connected\n[+] Connected clients: {len(self.connected_clients)}')
        return client_id

    def send_packet(self, data, client_socket: socket.socket):
        """
        Packs and sends a payload of data to a specified client.
        Args:
            data: payload of data to be sent.
            client_socket: socket used for the communication with a specific client.
        """
        try:
            client_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')

    def store_data(self, data, client_id):
        data = pickle.loads(data.split(b'<START>')[1].split(b'<END>')[0])
        header = list(data.keys())[0]
        if header != 'FINISH':
            query = f"""
            INSERT INTO training (client_id, {header}) VALUES (?, ?)
            ON CONFLICT (client_id) DO
            UPDATE SET {header} = ?
            """
            serialized_data = pickle.dumps(data[header])
            self.execute_query(query=query, values=(client_id, serialized_data, serialized_data))
            self.event_dict[header].set()
        else:
            self.event_dict[header].set()
            print('Fin event', epoch_finished.is_set())

    def transmit_initial_weights(self):
        pass
    
    def get_device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def give_labels(self, client_id):
        # Wait client's feature maps
        while True:
            if epoch_finished.is_set() == True:
                break
            features_recvd_event.wait()
            features_recvd_event.clear()
            query = 'SELECT model_outputs FROM training WHERE client_id = ?'
            client_features = pickle.loads(self.execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
            server_outputs = self.server_model(client_features)
            _, preds = torch.max(server_outputs, 1) 
            self.send_packet(data={'LABELS': preds}, client_socket=self.connected_clients[client_id][1])
        print('Finished training')


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

class ClientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5*5*256, out_features=10)
        
    def forward(self, x):
        x = self.fc1(torch.flatten(x, 1))
        return x

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1 = nn.Linear(3*3*512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv5(x))
        x = self.fc1(torch.flatten(x, 1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
# To execute, server_ip and server_port must be specified from the cl.
    if len(sys.argv) != 3:
        print('Incorrect number of command-line arguments.\nTo execute, server_ip and server_port must be specified from the cl.')
        sys.exit(1)

    server = Server(sys.argv[1], sys.argv[2])
    server.create_socket()
    server.create_db_schema()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    while len(server.connected_clients) < MIN_PARTICIPANTS:
        pass

    for id, (address, sock)  in server.connected_clients.items():
        print(f'[+] Labeling with client {id, address}')
        server.send_packet(data={'LABEL_EVENT': b''}, client_socket=sock)
        server.give_labels(id)
    
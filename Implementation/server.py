import socket
import threading
import pickle
import sys
import torch.nn as nn
import sqlite3
import torch
from fl_strategy import FL_Strategy
from fl_plan import FL_Plan
from client_models import ClientModel, ClientClassifier
from server_model import ServerModel
import time


# Events to ensure synchronization
features_recvd_event = threading.Event()
sufficient_clients_event = threading.Event()
labeling_finished = threading.Event()
label_lock = threading.Lock()
client_recvd_event = threading.Event()

# Constants
LABELING_CLIENT = -1

class Server:
    def __init__(self, server_ip, server_port):
        self.connected_clients = {}
        self.trained_clients = []
        self.labeled_clients = []
        self.ip = server_ip
        self.port = int(server_port)
        self.server_db_path = 'Implementation/server_data/server_db.db'
        self.event_dict = {'model_outputs': features_recvd_event, 'OK': client_recvd_event, 'final_model_outputs': labeling_finished}
        self.device = self.get_device()
        print(f'Using {self.device}')
        torch.manual_seed(32)
        self.server_model = ServerModel()
        self.server_model.to(device=self.device)
        torch.manual_seed(32)
        self.client_model = ClientModel()
        torch.manual_seed(32)
        self.classifier_model = ClientClassifier()
        self.recvd_initial_weights = 0

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
            model_updated_weights BLOB,
            classifier_updated_weights BLOB,
            data_size INT,
            FOREIGN KEY (client_id) REFERENCES clients (id)
        )
        """
        epoch_stats_table = """
        CREATE TABLE epoch_stats(
            epoch INT PRIMARY KEY,
            global_model BLOB,
            global_classifier BLOB,
            connected_clients INT,
            trained_clients INT
        )
        """
        self.execute_query(query=clients_table) if not self.check_table_existence(target_table='clients') else None
        self.execute_query(query=training_table) if not self.check_table_existence(target_table='training') else None
        self.execute_query(query=epoch_stats_table) if not self.check_table_existence(target_table='epoch_stats') else None
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
        exists = any(table[0] == target_table for table in tables) if tables is not None else False
        return exists
    
    def execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes a given query. Either for retrieval or update purposes.
        Args:
            query: Query to be executed
            values: Query values
            fetch_data_flag: Flag that signals a retrieval query
            fetch_all_flag: Flag that signals retrieval of all table data or just the first row.
        Returns:
            The data fetched for a specified query. If it is not a retrieval query then None is returned. 
        """
        try:
            connection = sqlite3.Connection(self.server_db_path)
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
                threading.Thread(target=self.listen_for_messages, args=(client_socket, client_id)).start()
                threading.Thread(target=self.give_labels, args=(client_id, client_socket)).start()
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
                        threading.Thread(target=self.handle_data, args=(data_packet, client_id)).start()
                        data_packet = b'' 
                if not data_chunk:
                    break
        except socket.error as error:
            # Handle client dropout
            global LABELING_CLIENT
            print(f'Error receiving data from {client_id, self.connected_clients[client_id][0]}:\n{error}')
            client_socket.close()
            self.connected_clients.pop(client_id)
            features_recvd_event.set() if LABELING_CLIENT == client_id else None
            self.labeled_clients.remove(client_id) if client_id in self.labeled_clients else None
    
    def handle_connections(self, client_address: tuple, client_socket: socket.socket):
        """
        When a client connects -> Add him on db if nonexistent, append to connected_clients list and transmit initial weights
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
        print(f'[+] Client {client_id, client_address} connected -> Connected clients: {len(self.connected_clients)}')
        #self.send_packet(data={'INITIAL_WEIGHTS': [self.client_model.state_dict(), self.classifier_model.state_dict()]}, client_socket=client_socket)
        self.send_packet(data={'PLAN': self.plan}, client_socket=client_socket)
        print(f'[+] Transmitted initial weights to client {client_id, client_address}')
        return client_id

    def send_packet(self, data: dict, client_socket: socket.socket):
        """
        Packs and sends a payload of data to a specified client.
        The format used is <START>DATA<END>, where DATA is a dictionary whose key is the header and whose value is the payload.
        Args:
            data: payload of data to be sent.
            client_socket: socket used for the communication with a specific client.
        """
        try:
            client_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')
            client_socket.close()

    def handle_data(self, data: dict, client_id: int):
        """
        Handles a received data packet according to its contents.
        A packet can be either be:
            1. Model Outputs during labeling process
            2. Updated model weights during training
            3. Signal that initial weights are received
        Args:
            data: Dictionary where the key is the header and the value is the payload
            client_id: The id of the sending client
        """
        # Get payload and header
        data = pickle.loads(data.split(b'<START>')[1].split(b'<END>')[0])
        header = list(data.keys())[0]
        if header.__contains__('model_outputs'):
            if header == 'model_outputs':
                query = f"""
                INSERT INTO training (client_id, model_outputs) VALUES (?, ?)
                ON CONFLICT (client_id) DO
                UPDATE SET model_outputs = ?
                """
                serialized_data = pickle.dumps(data[header])
                self.execute_query(query=query, values=(client_id, serialized_data, serialized_data))
            else:
                query = f"""
                INSERT INTO training (client_id, model_outputs, data_size) VALUES (?, ?, ?)
                ON CONFLICT (client_id) DO
                UPDATE SET model_outputs = ?, data_size = ?
                """
                serialized_outputs = pickle.dumps(data[header][0])
                data_size = data[header][1] * self.strategy.BATCH_SIZE
                self.execute_query(query=query, values=(client_id, serialized_outputs, data_size, serialized_outputs, data_size))
            self.event_dict['model_outputs'].set() if header == 'final_model_outputs' else None
        elif header == 'UPDATED_WEIGHTS':
            query = """
            INSERT INTO training (client_id, model_updated_weights, classifier_updated_weights) VALUES (?, ?, ?)
            ON CONFLICT (client_id) DO
            UPDATE SET model_updated_weights = ?, classifier_updated_weights = ? """
            serialized_model_weights = pickle.dumps(data[header][0])
            serialized_classifier_weights = pickle.dumps(data[header][1])
            self.execute_query(query=query, values=(client_id, serialized_model_weights, serialized_classifier_weights, serialized_model_weights, serialized_classifier_weights))
            self.trained_clients.append(client_id)
            print(f"\t[+] Received updated weights of client: {client_id, self.connected_clients[client_id][0]}")
        elif header == 'OK':
            self.recvd_initial_weights += 1
        self.event_dict[header].set() if header in self.event_dict.keys() else None


    def initialize_models(self):
        pass

    def initialize_strategy(self, config_file_path: str):
        """
        Initializes the FL Strategy and FL Plan objects based on the configuration file.
        Args:
            config_file_path: The path to the configuration file
        """
        self.strategy = FL_Strategy(config_file=config_file_path)
        self.plan = FL_Plan(epochs=self.strategy.GLOBAL_TRAINING_ROUNDS, lr=self.strategy.LEARNING_RATE,
                            loss=self.strategy.CRITERION, optimizer=self.strategy.OPTIMIZER, batch_size=self.strategy.BATCH_SIZE,
                            model_weights=self.client_model.state_dict(), classifier_weights=self.classifier_model.state_dict())
        print(f"[+] Emloyed Strategy:\n{self.strategy}")
        
    def get_device(self):
        """
        Check available devices (cuda or cpu). If cuda then cuda else cpu
        Returns: A torch.device() Object
        """
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def give_labels(self, client_id: int, client_socket: socket.socket):
        """
        Labeling process. Wait until features arrive, infer from them, send predictions until all data is labeled.
        Args:
            client_id: The client identifier
            client_socket: The client's socket
        """
        with label_lock:
            # Ensure that client has received weights
            global LABELING_CLIENT
            LABELING_CLIENT = client_id
            while not self.recvd_initial_weights > 0:
                pass
            self.recvd_initial_weights -= 1

            print(f'[+] Labeling with client {client_id, self.connected_clients[client_id][0]}')
            self.send_packet(data={'LABEL_EVENT': b''}, client_socket=client_socket)
            self.server_model.eval()
            with torch.inference_mode():
                while True:
                    # Ensure client's features are stored on the db
                    features_recvd_event.wait()
                    features_recvd_event.clear()
                    query = 'SELECT model_outputs FROM training WHERE client_id = ?'
                    client_features = pickle.loads(self.execute_query(query=query, values=(client_id, ), fetch_data_flag=True))
                    client_features.to(self.device)
                    server_outputs = self.server_model(client_features)
                    _, preds = torch.max(server_outputs, 1)
                    # Handle client that disconnected during labeling
                    try: 
                        self.send_packet(data={'LABELS': preds}, client_socket=self.connected_clients[client_id][1])
                    except KeyError as error:
                        break
                    # If this is the last batch if the client's data
                    if labeling_finished.is_set():
                        labeling_finished.clear()
                        break
        # Handle client that disconnected during labeling
        try:
            print(f'[+] Finished labeling with client {client_id, self.connected_clients[client_id][0]}')
            self.labeled_clients.append(client_id)
        except KeyError as error:
            print('[+] Client Disconnected.')


    def aggregate_global_model(self):
        """
        Aggregation process at the end of each global training round. Fetch necessary data and aggregate the global model by the Federated Averaging algorithm.\
        Returns:
            A list containing [The Federated Averaged client model, The Federated Averaged client classifier]
        """
        # Fetch the updated model weights of all trained clients
        query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.trained_clients) + ")"
        all_client_model_weights = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        # Fetch the updated classifier weights of all trained clients
        query = "SELECT classifier_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.trained_clients) + ")"
        all_client_classifier_weights = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        # Fetch the datasizes of all trained clients
        query = "SELECT data_size FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.trained_clients) + ")"
        datasizes = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        datasizes = [int(row[0]) for row in datasizes]
        return [self.federated_averaging(all_client_model_weights, datasizes), self.federated_averaging(all_client_classifier_weights, datasizes)]

    def federated_averaging(self, fetched_weights: list, fetched_datasizes: list):
        """
        Implementation of the federated averaging algotithm.
        Args:
            fetched_weights: The model weights of the participating/trained clients
            fetched_datasizes: The data sizes of the participating/trained clients
        Returns:
            avg_weights: The global model by Federated Averaging
        """
        # Dictionary for the global (averaged) weights
        avg_weights = {}
        # Calculate the data size of all the participating clients
        total_data = sum(datasize for datasize in fetched_datasizes)
        # For each client's updated weights
        for i in range(len(fetched_weights)):
            client_weight_dict = pickle.loads(fetched_weights[i][0])
            # For each layer of the model
            for key in client_weight_dict.keys():
                # Average the weights and normalize based on client's contribution to total data size
                if key in avg_weights.keys():
                    avg_weights[key] += client_weight_dict[key] * (fetched_datasizes[i] / total_data)
                else:
                    avg_weights[key] = client_weight_dict[key] * (fetched_datasizes[i] / total_data)
        return avg_weights

if __name__ == '__main__':
# To execute, server_ip and server_port must be specified from the cl.
    if len(sys.argv) != 3:
        print('Incorrect number of command-line arguments.\nTo execute, server_ip and server_port must be specified from the cl.')
        sys.exit(1)

    server = Server(sys.argv[1], sys.argv[2])
    server.create_socket()
    server.create_db_schema()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    server.initialize_strategy(config_file_path='Implementation/strategy_config.txt')
    # Wait for the minimum number of client to connect and label with the server
    while (len(server.connected_clients) < server.strategy.MIN_PARTICIPANTS_START) or (len(server.connected_clients) != len(server.labeled_clients)):
        pass
    
    for e in range(server.strategy.GLOBAL_TRAINING_ROUNDS):
        print(f'[+] Global training round {e + 1} initiated')
        query = "INSERT INTO epoch_stats (epoch, connected_clients) VALUES (?, ?)"
        server.execute_query(query=query, values=(e, len(server.connected_clients)))
        #transmit train signal to each client
        for client_id, (client_address, client_socket) in server.connected_clients.items():
            server.send_packet(data={'TRAIN': b''}, client_socket=client_socket)
        # Wait to receive model updates from the minimum number of clients to aggregate
        while len(server.trained_clients) < server.strategy.MIN_PARTICIPANTS_FIT:
            pass
        #aggregate global model
        aggr_models = server.aggregate_global_model()
        query = "UPDATE epoch_stats SET global_model = ?, global_classifier = ?, trained_clients = ? WHERE epoch = ?"
        server.execute_query(query=query, values=(pickle.dumps(aggr_models[0]), pickle.dumps(aggr_models[1]), len(server.trained_clients), e))
        #transmit global model
        for client_id, (client_address, client_socket) in server.connected_clients.items():
            server.send_packet(data={'AGGR_MODELS': aggr_models}, client_socket=client_socket)
        print(f'[+] Trasmitted global model to all participants')
        server.trained_clients.clear()
        time.sleep(5)
import socket
import threading
import pickle
import sys
import torch.nn as nn
import sqlite3


class Server:

    def __init__(self, server_ip, server_port):
        self.connected_clients = []
        self.trained_clients = []
        self.ip = server_ip
        self.port = int(server_port)
        self.server_db = 'Implementation/server_data/server_db.db'

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
        self.server_socket.listen()
        while True:
            client_socket, client_address = self.server_socket.accept()
            self.connected_clients.append(client_address)
            threading.Thread(target=server.listen_for_messages, args=(client_socket, )).start()
            print(f'[+] Client {client_address} connected')
            server.send_packet(data='hey', client_socket=client_socket)
            
    def listen_for_messages(self, client_socket):
        """
        Client-specific communication thread. Listens for incoming messages from a unique client.
        Args:
            client_socket: socket used from a particular client to establish communication.
        """
        data_packet = b''
        while True:
            try:
                data_chunk = client_socket.recv(4096)
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        data = data_packet.split(b'<START>')[1].split(b'<END>')[0]
                        print(pickle.loads(data))
                if not data_chunk:
                    break
            except socket.error as error:
                print(f'Error receiving data:\n{error}')

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


            


if __name__ == '__main__':
# To execute, server_ip and server_port must be specified from the cl.
    if len(sys.argv) != 3:
        print('Incorrect number of command-line arguments.\nTo execute, server_ip and server_port must be specified from the cl.')
        sys.exit(1)

    server = Server(sys.argv[1], sys.argv[2])
    server.create_socket()
    server.create_db_schema()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    
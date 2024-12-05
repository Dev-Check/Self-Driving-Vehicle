import socket
import cv2
import pickle
import struct
from tensorflow.keras.models import load_model
import numpy as np


model = load_model( r"yourmodelhere.keras")


server_ip = "your IP here" 
server_port = 12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)
print("Waiting for connection...")


client_socket, addr = server_socket.accept()
print("Connected to:", addr)

data = b""
payload_size = struct.calcsize("L")

while True:
    # Receive data from the client
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize the frame
    frame = pickle.loads(frame_data)

    # process frame
    resized_frame = cv2.resize(frame, (640, 480))  
    normalized_frame = resized_frame / 255.0
    input_array = np.expand_dims(normalized_frame, axis=0)

   
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction)

    
    client_socket.send(str(predicted_class).encode("utf-8"))
    print("Sent Prediction:", predicted_class)

client_socket.close()
server_socket.close()

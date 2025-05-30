import socket
import torch
import numpy as np
import joblib
from model import ShootingNet

model = ShootingNet()
model.load_state_dict(torch.load("shooting_model.pt"))
model.eval()

scaler = joblib.load("scaler.pkl")

HOST = "localhost"
PORT = 5001

def handle_connection(conn, addr):
    print(f"[PYTHON] Nowe połączenie z {addr}")
    with conn:
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    print("[PYTHON] Połączenie zakończone przez klienta.")
                    break

                values = list(map(float, data.decode().split(",")))
                situation = np.array(values).reshape(1, -1)
                situation_scaled = scaler.transform(situation)
                input_tensor = torch.tensor(situation_scaled, dtype=torch.float32)

                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = int(output.item() > 0.5)

                conn.sendall((str(prediction) + "\n").encode())
                print(f"[PYTHON] Decision sent: {prediction}")

            except Exception as e:
                print(f"[PYTHON] Błąd: {e}")
                break


# Serwer akceptujący wiele połączeń (np. z każdej nowej rundy)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[PYTHON] Listening on {HOST}:{PORT}...")

    while True:
        conn, addr = s.accept()
        handle_connection(conn, addr)

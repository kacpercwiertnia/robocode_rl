import socket
import numpy as np
import joblib

forest = joblib.load("forest_model.pkl")
scaler = joblib.load("forest_scaler.pkl")

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

                values = list(map(float, data.decode().strip().split(",")))
                situation = np.array(values).reshape(1, -1)
                situation_scaled = scaler.transform(situation)

                prediction = forest.predict(situation_scaled)[0]

                conn.sendall((str(int(prediction)) + "\n").encode())
                print(f"[FOREST] Decision sent: {int(prediction)}")

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

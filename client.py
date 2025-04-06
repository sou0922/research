import socket
import pickle
import torch
from model import Net
from utils import get_dataloaders, train_one_epoch, evaluate
import os

# クライアント識別（引数でID指定 or ホスト名から判定）
CLIENT_ID = int(input("Client ID (1/2/3): "))
PORT = 5000 + CLIENT_ID
HOST = ''  # すべてのNICで待ち受け

print(f"[Client {CLIENT_ID}] Listening on port {PORT}...")

# データローダの準備
train_loader, test_loader = get_dataloaders(CLIENT_ID)

# モデル受信用ソケット作成
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

while True:
    conn, addr = server_socket.accept()
    print(f"[Client {CLIENT_ID}] Connected from {addr}")

    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data += packet

    payload = pickle.loads(data)
    model_state = payload['model']
    rl_agent = payload['rl_agent']
    step = payload['step']

    # モデル初期化＆読み込み
    model = Net()
    model.load_state_dict(model_state)

    print(f"[Client {CLIENT_ID}] Training at step {step}...")
    train_one_epoch(model, train_loader)
    acc = evaluate(model, test_loader)
    print(f"[Client {CLIENT_ID}] Accuracy: {acc:.2f}%")

    # コントローラーへ送信準備
    return_data = {
        'model': model.state_dict(),
        'accuracy': acc,
        'rl_agent': rl_agent,
        'client_id': CLIENT_ID
    }

    # 接続先（コントローラー）を指定
    controller_ip = payload['controller_ip']
    controller_port = payload['controller_port']

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((controller_ip, controller_port))
        s.sendall(pickle.dumps(return_data))

    print(f"[Client {CLIENT_ID}] Sent results back to controller. Waiting for next task...\n")

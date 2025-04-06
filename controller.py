import socket
import pickle
import torch
from model import Net
from rl_agent import RLAgent
import time
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# クライアント情報（IPとポート）
clients = {
    1: (os.getenv("IP1"), 5001),
    2: (os.getenv("IP2"), 5002),
    # 3: (os.getenv("IP3"), 5003)
}

controller_ip = os.getenv("IP1")
controller_port = 6000

# コントローラ受信用ソケット
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((controller_ip, controller_port))
server_socket.listen(1)

# 初期化
model = Net()
rl_agent = RLAgent(num_clients=3)
current_client = 1
num_steps = 10
client_counts = {1: 0, 2: 0, 3: 0}

for step in range(num_steps):
    print(f"\n[Controller] Step {step+1}/{num_steps} | Sending to Client {current_client}")

    # モデル送信
    data_to_send = {
        'model': model.state_dict(),
        'rl_agent': rl_agent.q_table,
        'step': step,
        'controller_ip': controller_ip,
        'controller_port': controller_port
    }

    ip, port = clients[current_client]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, port))
        s.sendall(pickle.dumps(data_to_send))

    # 結果受信
    conn, addr = server_socket.accept()
    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet: break
        data += packet
    conn.close()

    result = pickle.loads(data)
    acc = result['accuracy']
    model.load_state_dict(result['model'])
    rl_agent.q_table = result['rl_agent']

    print(f"[Controller] Received accuracy {acc:.2f}% from Client {current_client}")

    # 強化学習エージェント更新 & 次クライアント選択
    rl_agent.update(current_client - 1, acc)
    current_client = rl_agent.select_action(current_client - 1) + 1
    client_counts[current_client] += 1

# モデル保存
torch.save(model.state_dict(), "final_model.pth")
print("\n[Controller] Training complete. Final model saved as final_model.pth")
print("[Controller] Client usage counts:", client_counts)

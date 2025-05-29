import argparse
import torch
import pickle
from model import SimpleCNN, SimpleAgent, SimpleAccAgent
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
import fcntl
import json

LOGPATH = "/mnt/log.txt"
gpu = 'cuda:3'
device = gpu if torch.cuda.is_available() else "cpu"

print(f"gpu: {device}", flush=True)

class Node:
    def __init__(self, name, ip, neighborsStr, max_hops):
        print(f"{name}が作成されました", flush=True)
        self.name = name
        self.ip = ip
        self.neibors = neighborsStr.split(",")
        self.dataPath = f"/tmp/{name}_cifar10.npz"
        self.prev_prev_node = None
        self.prev_node = None
        self.model = SimpleCNN()
        self.agent = SimpleAgent(decay_type="linear", decay_param=0.01, max_hops=max_hops)
        # self.agent = SimpleAccAgent(decay_type="linear", decay_param=0.01, max_hops=max_hops)

    def trainModel(self, device="cpu", epochs=1, batch_size=64):
        print(f"[DEBUG] trainModel() 開始: dataPath={self.dataPath}", flush=True)
        try:
            data = np.load(self.dataPath)
            images = data['images_train']
            labels = data['labels_train']
            images = images.astype(np.float32) / 255.0
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.astype(np.int64)
            dataset = torch.utils.data.TensorDataset(torch.tensor(images), torch.tensor(labels))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.model = self.model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                print(f"[DEBUG] Epoch {epoch+1}/{epochs} 開始", flush=True)
                for i, (x, y) in enumerate(loader):
                    try:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        out = self.model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        if (i+1) % 10 == 0 or (i+1) == len(loader):
                            print(f"[DEBUG] Epoch {epoch+1}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Batch {i+1} で例外発生: {e}", flush=True)
                        import traceback; traceback.print_exc()
                        raise  # 重大な場合はraiseして止める
                print(f"[Train] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}", flush=True)
        except Exception as e:
            print(f"[ERROR] trainModel() 全体で例外発生: {e}", flush=True)
            import traceback; traceback.print_exc()
        return self.model
    
    def testModel(self, device="cpu", when="学習時期"):
        data = np.load(self.dataPath)
        images = data['images_test']
        labels = data['labels_test']
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        labels = labels.astype(np.int64)
        dataset = torch.utils.data.TensorDataset(torch.tensor(images), torch.tensor(labels))
        # 前までbatch_size=64で学習していたが、テストは全データを一度に評価する
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        self.model = self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        log = f"[Test] {self.name} {when}精度:{acc*100:.2f}%"
        print(log, flush=True)
        self.appendLog(log)
        return acc
    
    def sendModelAgent(self, target_host, port=5000):
        candidates = [n for n in self.neibors if n != self.ip]
        if not candidates:
            print("[WARN] 送信先候補がありません", flush=True)
            return
        relay_info = {
            "prev_node": self.prev_node,
            "prev_prev_node": self.prev_prev_node
        }
        modelState = self.model.state_dict()
        relayBytes = pickle.dumps(relay_info)
        modelBytes = pickle.dumps(modelState)
        agentBytes = pickle.dumps(self.agent)
        startTime = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_host, port))

        # ヘッダを80バイト固定長で送信（時刻,モデルサイズ,エージェントサイズ）
        header_str = f"{startTime},{len(modelBytes)},{len(agentBytes)},{len(relayBytes)}"
        header_bytes = header_str.encode().ljust(80)

        s.sendall(header_bytes) # ヘッダ送信
        s.sendall(modelBytes)   # 本体送信
        s.sendall(agentBytes)   # エージェント送信
        s.sendall(relayBytes)   # リレー情報送信

        s.close()
        endTime = time.time()
        send_time = endTime - startTime
        log = f"[INFO] 送信完了 送信先:{target_host} ポート:{port} 送信時間:{send_time:.5f}秒"
        self.appendLog(log)

    def listen(self, port=5000):
        def recv_exact(conn, size):
            buf = b''
            while len(buf) < size:
                chunk = conn.recv(size - len(buf))
                if not chunk:
                    raise RuntimeError("Connection closed while receiving data")
                buf += chunk
            return buf

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', port))
        s.listen(5)
        print(f"{self.name} [INFO] Listening on port {port}...", flush=True)
        while True:
            try:
                conn, addr = s.accept()
                middleTime = time.time()
                print(f"[INFO] 接続完了 {addr}", flush=True)
                # ヘッダ受信
                header_bytes = recv_exact(conn, 80)
                header_str = header_bytes.decode().strip()
                start_time_str, model_size_str, agent_size_str, relay_size_str = header_str.split(',')

                start_time = float(start_time_str)
                model_size = int(model_size_str)
                agent_size = int(agent_size_str)
                relay_size = int(relay_size_str)
                total_size = model_size + agent_size
                # print(f"[DEBUG] start_time: {start_time}, model_size: {model_size}, agent_size: {agent_size}", flush=True)

                # 本体受信
                model_bytes = recv_exact(conn, model_size)
                agent_bytes = recv_exact(conn, agent_size)
                relay_bytes = recv_exact(conn, relay_size)
                conn.close()

                end_time = time.time()
                total_transfer_time = end_time - start_time
                receive_time = end_time - middleTime
                log = f"[INFO] 受信完了 受信元:{addr[0]} ポート:{addr[1]} 受信時間:{receive_time:.5f}秒"
                print(self.agent.step, self.agent.epsilon, flush=True)
                self.appendLog(log)

                model_state = pickle.loads(model_bytes)
                agent_state = pickle.loads(agent_bytes)
                relay_info = pickle.loads(relay_bytes)

                prev_prev_node = relay_info.get("prev_prev_node")
                prev_node = relay_info.get("prev_node")

                self.restoreModelAgent(model_state, agent_state)

                prev_accuracy = self.testModel(when = "学習前", device=device)
                # self.trainModel()
                try:
                    print("[DEBUG] trainModel() 開始", flush=True)
                    self.trainModel(device=device)
                    print("[DEBUG] trainModel() 終了", flush=True)
                except Exception as e:
                    print(f"[ERROR] trainModel() で例外発生: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    # 必要ならreturnやcontinueで処理を止める
                accuracy = self.testModel(when = "学習後", device=device)

                prev_state = (prev_prev_node, prev_node)
                state = (prev_node, self.ip)
                min_acc = float(self.get_log_line(5))  # 5行目が最低精度
                # 報酬=精度+通信時間
                reward = self.agent.calc_reward(prev_accuracy, accuracy, total_transfer_time)
                # 報酬=精度
                # reward = self.agent.calc_reward(prev_accuracy, accuracy, )
                self.agent.update(prev_state, self.ip, reward, state)
                action, choice = self.agent.get_action(state, self.neibors)
                epsilon = self.agent.decay_epsilon()

                log = f"[INFO] エージェント更新完了 前状態:{prev_state} 後状態:{state} 報酬:{reward} 次行動:{action} 要因:{choice} ランダム率:{epsilon * 100:.2f}%"
                self.appendLog(log)
                self.increment_transition_count(
                    transfer_time=total_transfer_time,
                    transfer_size=total_size,
                    target_node=self.name,
                    accuracy=accuracy  # ←追加
                    )
                
                # 位置更新
                self.prev_prev_node = prev_node
                self.prev_node = self.ip
                self.sendModelAgent(target_host=action, port=port)

            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
                traceback.print_exc()
                continue

    def restoreModelAgent(self, modelState, agentState):
        self.model.load_state_dict(modelState)
        self.agent = agentState
        print("[INFO] Model and agent params restored.", flush=True)

    def increment_transition_count(self, increment=1, transfer_time=0.0, transfer_size=0, target_node=None, accuracy=None):
        """
        1行目: 総遷移回数
        2行目: 総通信時間
        3行目: 総通信量（バイト）
        4行目: 各ノードへの遷移回数（例: {"h1": 3, "h2": 5, ...}）
        5行目: 最低精度
        6行目: 最大遷移時間
        """
        transition_counts = {}
        with open(LOGPATH, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            lines = f.readlines()
            count = int(lines[0].strip()) if len(lines) > 0 else 0
            total_time = float(lines[1].strip()) if len(lines) > 1 else 0.0
            total_size = int(lines[2].strip()) if len(lines) > 2 else 0
            if len(lines) > 3:
                try:
                    transition_counts = json.loads(lines[3].strip())
                except Exception:
                    transition_counts = {}
            min_acc = float(lines[4].strip()) if len(lines) > 4 else 1.0
            max_tr_time = float(lines[5].strip()) if len(lines) > 5 else 0.0

            # 値の更新
            count += increment
            total_time += transfer_time
            total_size += transfer_size

            # target_nodeが指定されていればカウントアップ
            if target_node:
                transition_counts[target_node] = transition_counts.get(target_node, 0) + 1

            # 最低精度の更新
            if accuracy is not None:
                min_acc = min(min_acc, accuracy)

            # 最大遷移時間の更新
            max_tr_time = max(max_tr_time, transfer_time)

            # 新しい内容を作成
            new_lines = [
                f"{count}\n",
                f"{total_time}\n",
                f"{total_size}\n",
                json.dumps(transition_counts) + "\n",
                f"{min_acc}\n",      # 5行目: 最低精度
                f"{max_tr_time}\n"   # 6行目: 最大遷移時間
            ]
            # 7行目以降があれば残す
            if len(lines) > 6:
                new_lines += lines[6:]

            f.seek(0)
            f.truncate()
            f.writelines(new_lines)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def appendLog(self, message):
        print(message, flush=True)
        with open(LOGPATH, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(message + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def get_log_line(self, n, logpath=LOGPATH):
        """
        log.txtからn行目（1始まり）を抜き出して返す関数
        """
        with open(logpath, "r") as f:
            lines = f.readlines()
            if 1 <= n <= len(lines):
                return lines[n-1].rstrip("\n")
            else:
                return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["listen", "create"], required=True)
    parser.add_argument("--name")
    parser.add_argument("--ip")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--neighborsStr", type=str, default="")
    parser.add_argument("--max_hops", type=int)
    args = parser.parse_args()

    # ノード作成
    node = Node(args.name, args.ip, args.neighborsStr, args.max_hops)

    if args.mode == "listen":
        node.listen()
    elif args.mode == "create":
        # 現状の精度
        prev_accuracy = node.testModel(when = "学習前", device=device)

        # CNN学習
        node.trainModel(device=device)

        # 学習後の制度
        accurecy = node.testModel(when = "学習後", device=device)

        # 遷移前
        state = (node.name, round(prev_accuracy, 2))
        action, log = node.agent.get_action(state, node.neibors)
        node.appendLog(log)

        print(f"[INFO] Initial action: {action}", flush=True)

        node.sendModelAgent(action, port=args.port)

if __name__ == "__main__":
    main()  
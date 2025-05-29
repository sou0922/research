from mininet.net import Mininet
from mininet.node import Controller
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.link import TCLink
from torchvision import datasets, transforms
import numpy as np
import os
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import json

# 各実験値
# ノードが8の場合
NODE = 8
DELAY_HOSTS = []
FLIP_HOSTS = ["h8"]
NOISE_HOSTS = []

# ノード数が16の場合
# NODE = 16
# DELAY_HOSTS = ["h1", "h2", "h3"]
# FLIP_HOSTS = ["h4", "h5", "h6"]
# NOISE_HOSTS = []

MAXHOPS = 200
SEED = 42
TIMEOUT = 2000
NOISE_LEVEL = 100
FLIP_RATE = 1.0 # 1.0で全て、0.2で20%だけflip
LOGPATH = "/mnt/log.txt"
PICNAME = "/mnt/graph.png"

def createNet(topology="default"):
    net = Mininet(link=TCLink)
    net.addController('c0')
    s1 = net.addSwitch('s1')
    hosts = []
    neighbors_dict = {}
    for i in range(1, NODE + 1):
        host_name = f'h{i}'
        host = net.addHost(host_name)
        # 遅延、パケットロス、帯域を設定
        if host_name in DELAY_HOSTS:
            net.addLink(host, s1 ,delay='20ms', loss=1, bw=100)
        else:
            net.addLink(host, s1)
        hosts.append(host)
    net.start()

    # スタート後にしかipアドレスは割り当てられない
    if topology == "random":
        for host in hosts:
            other_hosts = [h for h in hosts if h != host]
            random_hosts = random.sample(other_hosts, k=3)
            neighbors_dict[host.name] = random_hosts
    
    elif topology == "default":
        # ノードが8の場合
        neighbors_dict["h1"] = [net.get("h2"), net.get("h4"), net.get("h5")]
        neighbors_dict["h2"] = [net.get("h1"), net.get("h3"), net.get("h7")]
        neighbors_dict["h3"] = [net.get("h2"), net.get("h6"), net.get("h8")]
        neighbors_dict["h4"] = [net.get("h1"), net.get("h5"), net.get("h7")]
        neighbors_dict["h5"] = [net.get("h1"), net.get("h4"), net.get("h6")]
        neighbors_dict["h6"] = [net.get("h3"), net.get("h5"), net.get("h8")]
        neighbors_dict["h7"] = [net.get("h2"), net.get("h4"), net.get("h8")]
        neighbors_dict["h8"] = [net.get("h3"), net.get("h6"), net.get("h7")]

        # ノードが16の場合
        # neighbors_dict["h1"] = [net.get("h2"), net.get("h8"), net.get("h13")]
        # neighbors_dict["h2"] = [net.get("h1"), net.get("h6"), net.get("h8")]
        # neighbors_dict["h3"] = [net.get("h12"), net.get("h14"), net.get("h15")]
        # neighbors_dict["h4"] = [net.get("h6"), net.get("h7"), net.get("h10")]
        # neighbors_dict["h5"] = [net.get("h9"), net.get("h11"), net.get("h16")]
        # neighbors_dict["h6"] = [net.get("h2"), net.get("h4"), net.get("h7")]
        # neighbors_dict["h7"] = [net.get("h4"), net.get("h6"), net.get("h9")]
        # neighbors_dict["h8"] = [net.get("h1"), net.get("h2"), net.get("h9")]
        # neighbors_dict["h9"] = [net.get("h5"), net.get("h7"), net.get("h8")]
        # neighbors_dict["h10"] = [net.get("h4"), net.get("h11"), net.get("h12")]
        # neighbors_dict["h11"] = [net.get("h5"), net.get("h10"), net.get("h16")]
        # neighbors_dict["h12"] = [net.get("h3"), net.get("h10"), net.get("h13")]
        # neighbors_dict["h13"] = [net.get("h1"), net.get("h12"), net.get("h14")]
        # neighbors_dict["h14"] = [net.get("h3"), net.get("h13"), net.get("h15")]
        # neighbors_dict["h15"] = [net.get("h3"), net.get("h14"), net.get("h16")]
        # neighbors_dict["h16"] = [net.get("h5"), net.get("h11"), net.get("h15")]

    print(f"{neighbors_dict}")
    return net, hosts, neighbors_dict

def drawGraph(neighbors_dict, file_name=PICNAME):
    G = nx.Graph()
    for host_name, neighbors in neighbors_dict.items():
        for neighbor in neighbors:
            G.add_edge(host_name, neighbor.name)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=SEED)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    plt.title("Network Topology")
    plt.savefig(file_name)
    plt.close()

def distribute_cifar10(net):
    # CIFAR-10データセットを各ホストに分配（train/test分割）
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    images, labels = cifar10_dataset.data, np.array(cifar10_dataset.targets)
    num_hosts = len(net.hosts)
    images_per_host = len(images) // num_hosts
    for i, host in enumerate(net.hosts):
        start_idx = i * images_per_host
        end_idx = start_idx + images_per_host
        host_images = images[start_idx:end_idx]
        host_labels = labels[start_idx:end_idx]
        # 8:2でtrain/test分割
        n = len(host_images)
        n_train = int(n * 0.8)
        idx = np.random.permutation(n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        images_train = host_images[train_idx]
        labels_train = host_labels[train_idx]
        images_test = host_images[test_idx]
        labels_test = host_labels[test_idx]

        # ノイズを加える
        if host.name in NOISE_HOSTS:
            noise = np.random.normal(0, NOISE_LEVEL, images_train.shape).astype(np.int16)
            images_train = np.clip(images_train.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # テストデータはノイズを加えない
            # noise = np.random.normal(0, NOISE_LEVEL, images_test.shape).astype(np.int16)
            # images_test = np.clip(images_test.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if host.name in FLIP_HOSTS:
            n_flip = int(len(labels_train) * FLIP_RATE)
            flip_idx = np.random.choice(len(labels_train), n_flip, replace=False)
            for idx in flip_idx:
                orig = labels_train[idx]
                new_label = np.random.choice([l for l in range(10) if l != orig])
                labels_train[idx] = new_label

        np.savez(f"{host.name}_cifar10.npz", images_train=images_train, labels_train=labels_train, images_test=images_test, labels_test=labels_test)
        host.cmd(f"cp {host.name}_cifar10.npz /tmp/{host.name}_cifar10.npz")
        print(f"{host.name} に {n_train}枚のtrain画像, {n-n_train}枚のtest画像を配布しました。")

def resetLog(host_names):
    """
    ログファイルを初期化します。
    node_names: ["h1", "h2", ...] のようなノード名リスト
    """
    with open(LOGPATH, "w") as f:
        for _ in range(3):
            f.write("0\n")
        # 各ノードの遷移回数を0で初期化
        counts = {name: 0 for name in host_names}
        f.write(json.dumps(counts) + "\n")
        f.write("1.0\n")  # 4行目: 最低精度（初期値は1.0=100%）
        f.write("0.0\n")  # 5行目: 最大遷移時間（初期値は0.0秒）

def waitStop(net, maxHops=MAXHOPS, timeOut=5):
    startTime = time.time()
    while True:
        try:
            with open(LOGPATH, "r") as f:
                count = int(f.readline().strip())
        except Exception as e:
            count = 0
        if count >= maxHops:
            print(f"[INFO] 遷移回数が{maxHops}回に達したのでネットワークを停止します")
            break
        if time.time() - startTime > timeOut:
            print(f"[INFO] タイムアウト({timeOut}秒)のためネットワークを停止します")
            break
        print(count)
        time.sleep(1)
    net.stop()

def main():
    print("連合学習開始")
    # Mininetのログレベルを設定
    setLogLevel('info')

    # Mininetネットワークを作成
    net, hosts, neighbors_dict = createNet("default")
    host_names = [host.name for host in hosts]

    # ログファイルをリセット
    resetLog(host_names)

    # CIFAR-10データセットを各ホストに分配
    distribute_cifar10(net)

    # 開始ノード
    start_host = hosts[0]

    # ほかノードをリッスン状態
    for host in hosts:
        # if host != start_host:
        if True:
            neighborsStr = ",".join([h.IP() for h in neighbors_dict[host.name]])
            host.cmd(f"python3 -m node --mode listen --name {host.name} --ip {host.IP()} --neighborsStr {neighborsStr} --port 5000 --max_hops {MAXHOPS} > /tmp/{host.name}_node.log 2>&1 &")
            print(f"{host.name} がポート5000で待機中...")
    
    # 開始ノードで学習を開始
    neighborsStr = ",".join([h.IP() for h in neighbors_dict[start_host.name]])
    start_host.cmd(f"python3 -m node --mode create --name {start_host.name} --ip {start_host.IP()} --neighborsStr {neighborsStr} --port 5000 --max_hops {MAXHOPS} > /tmp/{start_host.name}_node.log 2>&1 &")

    # ホストの遷移回数をカウント
    waitStop(net, maxHops=MAXHOPS, timeOut=TIMEOUT)

    # ネットワークの形を保存
    drawGraph(neighbors_dict)

    # --- ここからログファイルを集めて保存 ---
    for host in hosts:
        logPath = f"/tmp/{host.name}_node.log"
        os.system(f"cp {logPath} ./{host.name}_node.log")
        print(f"{host.name} のログを ./{host.name}_node.log に保存しました。")

if __name__ == '__main__':
    main()
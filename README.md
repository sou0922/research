# Research
* 分散連合機械学習を行う
* 優秀なノード(良データ、良通信速度)のみで行う
* 優秀なノードを強化学習にて決定する

---

## Overview
気が向いたら軽く説明します

---

## Features
気が向いたら列挙します

---

## Requirement
気が向いたら列挙します
* huga 3.5.2
* hogehuga 1.0.2

---

## Installation
気が向いたら列挙します

```bash
pip install huga_package
```

---

## Usage
アクティベート
```bash
source ../../bin/activate
```
ビルド
```bash
docker build -t mininet-custom .
```
起動(without GPU)
```bash
docker run --rm -it --privileged -v $(pwd):/mnt mininet-custom
```
起動(with GPU)
```bash
docker run --rm -it --privileged --gpus all -v $(pwd):/mnt mininet-custom
```
ディレクトリ移動
```bash
cd mnt
```
実行
```bash
python3 main.py
```

---

## Note
注意点などが残します

---

## Author
* So Tamaki
* E-mail

---

## License
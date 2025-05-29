#!/bin/bash

# 初期化ディレクトリの作成
mkdir -p /var/run/openvswitch
mkdir -p /etc/openvswitch

# conf.dbが存在しない場合のみ初期化（2回目以降の起動用）
if [ ! -f /etc/openvswitch/conf.db ]; then
    ovsdb-tool create /etc/openvswitch/conf.db /usr/share/openvswitch/vswitch.ovsschema
fi

# Open vSwitch デーモン起動
ovsdb-server --remote=punix:/var/run/openvswitch/db.sock \
             --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
             --pidfile --detach

ovs-vsctl --no-wait init
ovs-vswitchd --pidfile --detach

# シェルに入る
exec /bin/bash


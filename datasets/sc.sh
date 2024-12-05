#!/bin/bash

# 遍历当前目录及所有子目录中的 band.conf 文件
find . -type f -name 'band.conf' | while read band_conf_path; do
    # 检查文件中是否缺少 DIM = 行
    if ! grep -q "DIM =" "$band_conf_path"; then
        echo "Missing DIM = in $band_conf_path"
    fi
done

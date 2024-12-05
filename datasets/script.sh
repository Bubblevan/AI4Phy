#!/bin/bash

# 遍历当前目录下的所有子目录
find . -type d | while read dir; do
    band_conf_path="$dir/band.conf"
    # 检查是否存在 band.conf 文件
    if [[ -f "$band_conf_path" ]]; then
        # 检查文件中是否已包含指定的行
        if ! grep -Fxq "FULL_FORCE_CONSTANTS = .TRUE." "$band_conf_path"; then
            # 确保文件以换行符结束
            tail -c1 "$band_conf_path" | read -r _ || echo >> "$band_conf_path"
            # 添加FULL_FORCE_CONSTANTS = .TRUE.到新的一行
            echo "FULL_FORCE_CONSTANTS = .TRUE." >> "$band_conf_path"
            echo "Added line to $band_conf_path"
        else
            echo "Line already exists in $band_conf_path"
        fi
        # 切换到包含 band.conf 的目录
        cd "$dir"
        # 执行 phonopy 命令
        phonopy -p band.conf
        # 返回到脚本所在的原始目录
        cd - > /dev/null
    fi
done

#!/bin/bash

# 设置 requirements 文件名
REQ_FILE="requirements.txt"

# 逐行读取 requirements.txt
while IFS= read -r package || [ -n "$package" ]; do
  # 跳过空行和注释
  [[ -z "$package" || "$package" == \#* ]] && continue

  echo "🔵 尝试使用 mamba 安装: $package"
  if mamba install -y "$package"; then
    echo "✅ 成功使用 mamba 安装: $package"
  else
    echo "⚠️ mamba 安装失败，尝试使用 pip 安装: $package"
    if pip install "$package"; then
      echo "✅ 成功使用 pip 安装: $package"
    else
      echo "❌ pip 安装失败，跳过: $package"
    fi
  fi
done < "$REQ_FILE"
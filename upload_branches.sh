#!/bin/bash

# 设置你的目标 GitHub 仓库地址（请改成你的新仓库）
REMOTE_URL="https://github.com/jiameng-ji/LifeSignalV2.git"

# 要处理的所有分支名
branches=(
  main
  aifeatures
  improvement
  machinelearning
  modeltraining
  newmodel
  watch
)

# 当前目录路径
SOURCE_DIR=$(pwd)

for branch in "${branches[@]}"; do
  echo "=== 处理分支: $branch ==="

  # 切换到远程分支
  git checkout origin/$branch

  # 创建对应的目录
  TARGET_DIR="../copy-$branch"
  mkdir -p "$TARGET_DIR"

  # 拷贝项目内容（排除 .git）
  rsync -av --exclude='.git' ./ "$TARGET_DIR"

  # 初始化并推送到新仓库
  cd "$TARGET_DIR"
  git init
  git checkout -b "$branch"
  git add .
  git commit -m "Initial commit from $branch"
  git remote add origin "$REMOTE_URL"
  git push -u origin "$branch"

  # 回到源目录
  cd "$SOURCE_DIR"
done

echo "✅ 所有分支复制上传完成！"


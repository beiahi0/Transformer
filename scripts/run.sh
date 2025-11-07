#!/bin/bash

# 运行基础实验
# 确保设置随机种子 
python src/train.py \
    data=iwslt2017 \
    model=transformer \
    trainer.max_epochs=20 \
    seed=42 \
    trainer.gradient_clip_val=1.0 # 增加梯度裁剪

🚀 运行与复现
本项目使用 PyTorch Lightning 和 Hydra 进行配置管理和训练。

## 硬件要求
GPU: 强烈推荐使用 NVIDIA GPU 进行训练。

GPU 显存 (VRAM): 我们的模型（d_model=128, 2层）非常小，在 batch_size=32 时，>= 6GB 显存即可满足训练需求（推荐 8GB+）。

系统内存 (RAM): >= 16GB（用于加载和预处理 IWSLT 2017 数据集）。

##  环境设置
克隆本仓库：


```bash

git clone https://github.com/beiahi0/Transformer.git
cd Transformer
```

安装本项目特定的依赖：

```bash
pip install -r requirements.txt
```
下载 Spacy 语言模型（用于分词）：


```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```
##  复现训练 (Training)
我们所有的实验参数（batch_size=32, d_model=128, lr=3e-4 等）都已在 configs/ 目录中定义。config.yaml 中设置的随机种子为 42。

要精确复现我们的训练结果（约 60 个 epoch），请运行以下命令。该命令将：

使用 seed=42。

启用 deterministic=True 以确保 CUDA 算法的可复现性。

自动下载 IWSLT 2017 数据集并构建词表。

使用 WandbLogger 记录日志（请确保你已登录 wandb）。


# 运行训练
```bash
python src/train.py 
```
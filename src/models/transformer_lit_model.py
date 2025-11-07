# src/models/transformer_lit_model.py
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 示例调度器
from typing import Optional  # <--- 添加这个 import

from torchmetrics.text import BLEUScore

# 导入我们刚创建的模型
from src.models.components.transformer_core import Transformer


class TransformerLitModel(LightningModule):
    def __init__(
        self,
        d_model: int = 128,  # 对应作业中的超参数
        n_layers: int = 2,  #
        n_heads: int = 4,  #
        d_ff: int = 512,  #
        dropout: float = 0.1,
        lr: float = 3e-4,  #
        max_len: int = 5000,
    ):
        super().__init__()
        # 保存超参数，自动记录
        self.save_hyperparameters()

        # 约定 PAD index
        self.pad_idx = 1  # 对应 IWSLTDataModule.PAD_IDX

        # self.model = Transformer(
        #     src_vocab_size=src_vocab_size,
        #     tgt_vocab_size=tgt_vocab_size,
        #     d_model=d_model,
        #     n_layers=n_layers,
        #     n_heads=n_heads,
        #     d_ff=d_ff,
        #     dropout=dropout,
        # )

        # 损失函数，忽略 padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.val_bleu = BLEUScore(n_gram=4, smooth=True)
        self.test_bleu = BLEUScore(n_gram=4, smooth=True)

    def setup(self, stage: str):
        # 从 self.trainer (由 Lightning 自动附加) 访问 datamodule
        src_vocab_size = self.trainer.datamodule.src_vocab_size
        tgt_vocab_size = self.trainer.datamodule.tgt_vocab_size

        # 现在，使用 hparams 和 vocab_sizes 构建真正的模型
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.hparams.d_model,
            n_layers=self.hparams.n_layers,
            n_heads=self.hparams.n_heads,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            max_len=self.hparams.max_len,
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

    def _calculate_loss(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]

        # "Teacher Forcing"：目标向右移动一位
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 创建掩码
        # (注意：我们的模型期望 (N, 1, 1, src_seq_len) 和 (N, 1, tgt_seq_len, tgt_seq_len))
        src_mask = self.model.create_src_mask(src, self.pad_idx)
        tgt_mask = self.model.create_tgt_mask(tgt_input, self.pad_idx)

        # 模型预测
        preds = self.forward(src, tgt_input, src_mask, tgt_mask)

        # preds: (N, seq_len, vocab_size) -> (N * seq_len, vocab_size)
        # tgt_output: (N, seq_len) -> (N * seq_len)
        loss = self.criterion(
            preds.reshape(-1, preds.size(-1)), tgt_output.reshape(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/perplexity", torch.exp(loss), on_step=False, on_epoch=True
        )

        datamodule = self.trainer.datamodule
        vocab_en = datamodule.vocab_en
        bos_idx = datamodule.BOS_IDX
        eos_idx = datamodule.EOS_IDX
        pad_idx = datamodule.PAD_IDX

        # 准备输入
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = self.model.create_src_mask(src, self.pad_idx)

        # 使用贪心解码生成预测
        pred_indices = self.greedy_decode(src, src_mask, max_len=50)

        # 将预测的索引解码为字符串
        pred_strings = self._decode_batch_to_strings(
            pred_indices, vocab_en, bos_idx, eos_idx, pad_idx
        )

        # 将目标的索引解码为字符串
        # 注意：torchmetrics 的 BLEU 期望参考是一个 "list of lists"
        ref_strings_list = [
            [s]
            for s in self._decode_batch_to_strings(
                tgt, vocab_en, bos_idx, eos_idx, pad_idx
            )
        ]

        # 更新 BLEU 分数
        self.val_bleu.update(pred_strings, ref_strings_list)

        # 在 epoch 结束时记录 BLEU
        self.log(
            "val/bleu",
            self.val_bleu,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test/perplexity", torch.exp(loss), on_step=False, on_epoch=True
        )

        # 2. 计算 BLEU (逻辑同 validation_step)
        datamodule = self.trainer.datamodule
        vocab_en = datamodule.vocab_en
        bos_idx = datamodule.BOS_IDX
        eos_idx = datamodule.EOS_IDX
        pad_idx = datamodule.PAD_IDX

        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = self.model.create_src_mask(src, self.pad_idx)

        pred_indices = self.greedy_decode(src, src_mask, max_len=50)

        pred_strings = self._decode_batch_to_strings(
            pred_indices, vocab_en, bos_idx, eos_idx, pad_idx
        )
        ref_strings_list = [
            [s]
            for s in self._decode_batch_to_strings(
                tgt, vocab_en, bos_idx, eos_idx, pad_idx
            )
        ]

        self.test_bleu.update(pred_strings, ref_strings_list)
        self.log(
            "test/bleu",
            self.test_bleu,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    # --- !!! 新增方法：贪心解码 (Greedy Decode) !!! ---
    def greedy_decode(self, src, src_mask, max_len: int):
        """
        用于推理的贪心解码函数
        src: (N, src_seq_len)
        src_mask: (N, 1, 1, src_seq_len)
        """
        # (我们假设已经从 datamodule 获取了 BOS/EOS)
        bos_idx = 2
        eos_idx = 3

        bos_idx = self.trainer.datamodule.BOS_IDX
        eos_idx = self.trainer.datamodule.EOS_IDX

        N = src.size(0)

        # 1. Encoder 只运行一次
        encoder_output = self.model.encoder(
            src, src_mask
        )  # (N, src_seq_len, d_model)

        # 2. Decoder 输入以 <bos> token 开始
        # (N, 1)
        tgt = torch.full((N, 1), bos_idx, dtype=torch.long, device=self.device)

        # 3. 循环生成
        for _ in range(max_len - 1):  # 减去 <bos>
            # 创建 target mask
            tgt_mask = self.model.create_tgt_mask(
                tgt, self.pad_idx
            )  # (N, 1, seq_len, seq_len)

            # 运行 Decoder
            # (N, seq_len, d_model)
            decoder_output = self.model.decoder(
                tgt, encoder_output, src_mask, tgt_mask
            )

            # (N, seq_len, vocab_size)
            logits = self.model.final_linear(decoder_output)

            # 只关心最后一个词的 logits
            # (N, vocab_size)
            last_logits = logits[:, -1, :]

            # 贪心选择 (N, 1)
            pred_token = torch.argmax(last_logits, dim=-1).unsqueeze(1)

            # 将新预测的 token 拼接到 tgt
            # (N, seq_len + 1)
            tgt = torch.cat([tgt, pred_token], dim=1)

            # 简单停止条件：如果一个 batch 中的所有句子都生成了 <eos> (效率不高, 但可行)
            if (pred_token.squeeze() == eos_idx).all():
                break

        return tgt

    # --- !!! 新增方法：解码为字符串 !!! ---
    def _decode_batch_to_strings(
        self, idx_tensor, vocab, bos_idx, eos_idx, pad_idx
    ):
        """
        辅助函数，将一个 batch 的索引 tensor 解码为字符串列表。
        idx_tensor: (N, seq_len)
        """
        special_indices = {bos_idx, eos_idx, pad_idx}
        itos = vocab.get_itos()  # 获取 index -> string 的映射

        decoded_strings = []

        for indices in idx_tensor:  # 遍历 batch
            sentence = []
            for idx in indices:
                idx_item = idx.item()

                # 跳过 <bos> 和 <pad>
                if idx_item == bos_idx or idx_item == pad_idx:
                    continue

                # 遇到 <eos> 就停止
                if idx_item == eos_idx:
                    break

                sentence.append(itos[idx_item])

            decoded_strings.append(" ".join(sentence))

        return decoded_strings

    def configure_optimizers(self):
        """
        实现 AdamW (进阶要求) 和 学习率调度 (进阶要求)
        """
        # 作业推荐 Adam, 进阶要求 AdamW
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,  # 5 个 epoch 验证集损失没有下降则降低学习率
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # 监控验证集损失
            },
        }

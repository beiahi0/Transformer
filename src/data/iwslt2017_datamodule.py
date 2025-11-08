import spacy
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Iterable

from torch.nn.utils.rnn import pad_sequence
import torch


def load_tokenizers():
    """加载 Spacy 分词器"""
    try:
        spacy_de = spacy.load("de_core_news_sm")
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        print("Spacy 模型 'de_core_news_sm' 或 'en_core_web_sm' 未找到。")
        print("请运行:")
        print("python -m spacy download de_core_news_sm")
        print("python -m spacy download en_core_web_sm")
        raise
    return spacy_de, spacy_en


def tokenize(text: str, tokenizer) -> List[str]:
    """辅助函数：对文本进行分词"""
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(
    data_iter: Iterable, tokenizer, language: str
) -> Iterable[List[str]]:
    """一个辅助的生成器函数，用于 build_vocab_from_iterator"""
    for data in data_iter:
        yield tokenize(data["translation"][language], tokenizer)


class IWSLTDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 定义特殊的 token
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

        # 初始化
        self.spacy_de, self.spacy_en = load_tokenizers()
        self.vocab_de = None
        self.vocab_en = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """下载数据集 (只在 1 个 GPU 上运行)"""
        print("Downloading IWSLT2017 dataset...")
        load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="train",
        )
        load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="validation",
        )
        load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="test",
        )
        print("Download complete.")

    def setup(self, stage: str = None):
        """加载数据、构建词表 (在所有 GPU 上运行)"""
        print("Setting up DataModule...")
        train_data = load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="train",
        )
        val_data = load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="validation",
        )
        test_data = load_dataset(
            "IWSLT/iwslt2017",
            "iwslt2017-en-de",
            split="test",
        )

        # 构建词表 (Vocab)
        print("Building vocabularies...")

        # 德语 (源语言, de) 词表
        train_iter_de = yield_tokens(train_data, self.spacy_de, "de")
        self.vocab_de = build_vocab_from_iterator(
            train_iter_de,
            min_freq=2,
            specials=self.special_symbols,
            special_first=True,
        )
        self.vocab_de.set_default_index(self.UNK_IDX)

        # 英语 (目标语言, en) 词表
        train_iter_en = yield_tokens(train_data, self.spacy_en, "en")
        self.vocab_en = build_vocab_from_iterator(
            train_iter_en,
            min_freq=2,
            specials=self.special_symbols,
            special_first=True,
        )
        self.vocab_en.set_default_index(self.UNK_IDX)

        print("Vocabularies built.")
        print(f"DE (source) vocab size: {self.src_vocab_size}")
        print(f"EN (target) vocab size: {self.tgt_vocab_size}")

        # 分配数据集
        self.data_train = train_data
        self.data_val = val_data
        self.data_test = test_data
        print("Data setup complete.")

    # --- 修复后的属性 ---
    @property
    def src_vocab_size(self) -> int:
        """获取源语言 (德语) 词汇表大小"""
        if not self.vocab_de:
            self.setup()
        return len(self.vocab_de)

    @property
    def tgt_vocab_size(self) -> int:
        """获取目标语言 (英语) 词汇表大小"""
        if not self.vocab_en:
            self.setup()
        return len(self.vocab_en)

    def collate_fn(self, batch):
        """
        处理一个 batch 的数据：分词, 添加 BOS/EOS, 数值化, Padding
        !!! 这一版返回 batch_first=True (N, seq_len) !!!
        """
        src_batch, tgt_batch = [], []

        for sample in batch:
            src_text = sample["translation"]["de"]  # 源: 德语
            tgt_text = sample["translation"]["en"]  # 目标: 英语

            src_tokens = tokenize(src_text, self.spacy_de)
            tgt_tokens = tokenize(tgt_text, self.spacy_en)

            src_indices = self.vocab_de(src_tokens)
            tgt_indices = self.vocab_en(tgt_tokens)

            src_tensor = torch.tensor(
                [self.BOS_IDX] + src_indices + [self.EOS_IDX], dtype=torch.long
            )
            tgt_tensor = torch.tensor(
                [self.BOS_IDX] + tgt_indices + [self.EOS_IDX], dtype=torch.long
            )

            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)

        src_padded = pad_sequence(
            src_batch, batch_first=True, padding_value=self.PAD_IDX
        )
        tgt_padded = pad_sequence(
            tgt_batch, batch_first=True, padding_value=self.PAD_IDX
        )

        # 返回一个字典，这对于 LightningModule 更友好
        return {
            "src": src_padded,  # (N, src_seq_len)
            "tgt": tgt_padded,  # (N, tgt_seq_len)
        }

    # --- DataLoaders ---
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

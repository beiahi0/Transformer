# src/data.py (续)
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

# 特殊标记
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"  # Start of Sentence
EOS_TOKEN = "[EOS]"  # End of Sentence


def load_data():
    # 加载 IWSLT2017 数据集，选择 'en-de' 配置
    dataset = load_dataset(
        "IWSLT/iwslt2017", "iwslt2017-en-de", trust_remote_code=True
    )

    # 数据集包含 'train', 'validation', 'test'
    print(dataset)
    print("训练集示例:", dataset["train"][0])
    return dataset


def get_all_sentences(dataset, lang):
    """迭代器，用于获取所有句子"""
    for split in dataset.keys():
        for example in dataset[split]:
            yield example["translation"][lang]


def build_tokenizer(dataset, lang, vocab_size=30000):
    """训练并返回一个分词器"""
    tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN],
    )

    # 使用数据集训练分词器
    tokenizer.train_from_iterator(
        get_all_sentences(dataset, lang), trainer=trainer
    )

    # 将 PAD_TOKEN 的 ID 设置为 0，方便后续 padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN
    )

    return tokenizer


# 在主函数中调用
if __name__ == "__main__":
    dataset = load_data()

    print("构建英语分词器...")
    tokenizer_en = build_tokenizer(dataset, "en")
    tokenizer_en.save("tokenizer-en.json")

    print("构建德语分词器...")
    tokenizer_de = build_tokenizer(dataset, "de")
    tokenizer_de.save("tokenizer-de.json")

    print("分词器构建完毕并保存。")

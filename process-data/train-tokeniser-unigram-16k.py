from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


tokenizer = Tokenizer(Unigram())
trainer = UnigramTrainer(vocab_size=16000, special_tokens=["<|endoftext|>"], unk_token="<|endoftext|>")
tokenizer.pre_tokenizer = Whitespace()
files = ["../data/babylm_data/babylm_10M/train_full"]
tokenizer.train(files, trainer)
# tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=256)
# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:0 [SEP]:0",
#     special_tokens=[
#         ("[CLS]", tokenizer.token_to_id("[CLS]")),
#         ("[SEP]", tokenizer.token_to_id("[SEP]")),
#     ],
# )
tokenizer.save("tokeniser-unigram-16000.json")

from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from transformers import BertTokenizer, XLNetTokenizer, QuestionAnsweringPipeline, XLNetConfig, XLNetModel
import tensorflow as tf


# My arbitrary sentence
sentence = "We need small heroes so that big heroes can shine"
# Bert vocabularies
bertBaseCased = "/home/sabur/Downloads/TweetQAexperiments/bert-base-cased-vocab.txt"
bertBaseUncased = "/home/sabur/Downloads/TweetQAexperiments/bert-base-uncased-vocab.txt"
bertLargeCased = "/home/sabur/Downloads/TweetQAexperiments/bert-large-cased-vocab.txt"
bertLargeUncased = "/home/sabur/Downloads/TweetQAexperiments/bert-large-uncased-vocab.txt"
# GPT-2 vocabularies
gpt2Vocab = "gpt2-vocab.json"
gpt2LargeVocab = "gpt2-large-vocab.json"
# Instantiate a Bert tokenizers
WordPiece = BertWordPieceTokenizer(bertLargeUncased)
WordPieceEncoder = WordPiece.encode(sentence)
# Print the ids, tokens and offsets
print(WordPieceEncoder.ids)
print(WordPieceEncoder.tokens)
print(WordPieceEncoder.offsets)



tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = QuestionAnsweringPipeline.from_pretrained('xlnet-base-cased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :] # Batch size 1
outputs = model(input_ids)
start_scores, end_scores = outputs[:2]


trans = BertTokenizer(bertLargeUncased, do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', tokenize_chinese_chars=True)
print(trans.vocab)
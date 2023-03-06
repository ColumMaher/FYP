import torch
import string
import math
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM

roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModelForMaskedLM.from_pretrained("roberta-base").eval()

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()

top_k = 10

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + "[PAD]"
    tokens = []
    for w in pred_idx:
        token = "".join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace("##", ""))
    return "\n".join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace("[MASK]", tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += " ."

    input_ids = torch.tensor(
        [tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)]
    )
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx



train_iter = PennTreebank(split="train")
tokenizer = get_tokenizer("basic_english")
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def predict_next_word(
    model: nn.Module,
    prompt: str,
    vocab: torchtext.vocab.Vocab,
    top_k: int,
    temperature: float = 1.0,
) -> str:
    """
    Generates the next word in a sentence given the prompt and the trained model.

    Args:
        model: nn.Module, the trained LSTM model
        prompt: str, the prompt to generate the next word for
        vocab: torchtext.vocab.Vocab, the vocabulary used during training
        top_k: int, the number of next most probably words
        temperature: float, the sampling temperature to use when generating the next word

    Returns:
        list of tuple, the generated next word and its probability
    """
    # Tokenize the prompt and convert to tensor
    tokenized_prompt = tokenizer(prompt)
    tensor_prompt = torch.tensor(vocab(tokenized_prompt), dtype=torch.long)

    # Pass the prompt through the model
    with torch.no_grad():
        model.eval()  # turn on evaluation mode
        output = model(tensor_prompt.unsqueeze(0))
        output = output[-1, :, :]

    # Sample the next word from the output
    output = output / temperature
    probs = F.softmax(output, dim=-1)
    top_k_probs, top_k_words = probs.topk(k=top_k)
    next_word_list = [
        (vocab.get_itos()[i[0].item()], p[0].item())
        for i, p in zip(top_k_words, top_k_probs)
    ]
    return next_word_list


class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                    dropout=dropout_rate, batch_first=True)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear = nn.Linear(hidden_dim, vocab_size)

    if tie_weights:
          assert embedding_dim == hidden_dim, 'cannot tie, check dims'
          self.linear.weight = self.embedding.weight
    self.init_weights()

  def forward(self, x):
    # x is a batch of input sequences
    x = self.embedding(x)
    x, _ = self.lstm(x)
    x = self.linear(x)
    return x

  def init_weights(self):
    init_range_emb = 0.1
    init_range_other = 1/math.sqrt(self.hidden_dim)
    self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
    self.linear.weight.data.uniform_(-init_range_other, init_range_other)
    self.linear.bias.data.zero_()
    for i in range(self.num_layers):
        self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                self.hidden_dim).uniform_(-init_range_other, init_range_other)
        self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
                self.hidden_dim).uniform_(-init_range_other, init_range_other)

  def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 100
num_layers = 2
dropout_rate = 0.4
tie_weights = True
lstm_model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights)
lstm_model.load_state_dict(torch.load("C:/Users/Colum/Documents/CS4125Repo/cs4125_project/FYP/Models/LSTM_Model.pt"))
lstm_model.eval()


class TransformerModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, num_head: int, hidden_dim: int,
                 num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_head, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, vocab_size]
        """
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

embedding_dim = 200  # embedding dimension
hidden_dim = 200  # dimension of the feedforward network model in nn.TransformerEncoder
num_layers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
num_head = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2
transformer_model = TransformerModel(vocab_size, embedding_dim, num_head, hidden_dim, num_layers, dropout)
bptt = 35

def transformer_predict_next_word(model: nn.Module, prompt: str, vocab: torchtext.vocab.Vocab, top_k: int,
                      temperature: float = 1.0) -> str:
    """
    Generates the next word in a sentence given the prompt and the trained model.

    Args:
        model: nn.Module, the trained model
        prompt: str, the prompt to generate the next word for
        vocab: torchtext.vocab.Vocab, the vocabulary used during training
        top_k: int, the number of next most probably words
        temperature: float, the sampling temperature to use when generating the next word

    Returns:
        list of tuple, the generated next word and its probability
    """
    # Tokenize the prompt and convert to tensor
    tokenized_prompt = tokenizer(prompt)
    tensor_prompt = torch.tensor(vocab(tokenized_prompt), dtype=torch.long)
    tensor_prompt = tensor_prompt
    src_mask = generate_square_subsequent_mask(1)

    seq_len = tensor_prompt.size(0)
    if seq_len != bptt:
        src_mask = src_mask[:seq_len, :seq_len]

    # Pass the prompt through the model
    with torch.no_grad():
        model.eval()  # turn on evaluation mode
        output = model(tensor_prompt.unsqueeze(0), src_mask)
        output = output[-1, :, :]

    # Sample the next word from the output
    output = output / temperature
    probs = F.softmax(output, dim=-1)
    top_k_probs, top_k_words = probs.topk(k=top_k)
    next_word_list = [(vocab.get_itos()[i[0].item()], p[0].item()) for i, p in zip(top_k_words, top_k_probs)]
    return next_word_list


def get_model_predictions(text_sentence, top_clean=5):
    top_k = 10

    LSTM = predict_next_word(lstm_model, text_sentence, vocab, top_k=top_k)
    LSTM_Pred = ""
    for s, n in LSTM:
        LSTM_Pred += s + "\n"

    TF = transformer_predict_next_word(transformer_model, text_sentence, vocab, top_k=top_k)
    TF_Pred = ""
    for s,n in TF:
        TF_Pred += s + "\n"

    text_sentence = text_sentence + " [MASK]"

    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    input_ids2, mask_idx2 = encode(roberta_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
        predict2 = roberta_model(input_ids2)[0]
    bert_prediction = decode(
        bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean
    )
    roberta_prediction = decode(
        roberta_tokenizer, predict2[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean
    )

    return {'lstm': LSTM_Pred,
            'transformer': TF_Pred,
            'bert': bert_prediction,
            'roberta': roberta_prediction
            }


print(get_model_predictions("How are you", 5))
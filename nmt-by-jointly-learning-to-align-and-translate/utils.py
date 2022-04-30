# Import Libraries
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from math import exp

# Pre-defined tokens
UNK_TOKEN = '<unk>' # token representing "unknown word"
PAD_TOKEN = '<pad>' # token representing "padding"
SOS_TOKEN = '<sos>' # token representing "start of sentence"
EOS_TOKEN = '<eos>' # token representing "end of sentence"

src_tokenizer = get_tokenizer(tokenizer='spacy', language='de')
trg_tokenizer = get_tokenizer(tokenizer='spacy', language='en')

def yield_src_tokens(data):
  for src_sentence, _ in data:
    yield src_tokenizer(src_sentence.strip().lower())

def yield_trg_tokens(data):
  for _, trg_sentence in data:
    yield trg_tokenizer(trg_sentence.strip().lower())

# Build a vocabulary for German
vocab_data = Multi30k(split=('train'), language_pair=('de', 'en'))
src_vocab = build_vocab_from_iterator(yield_src_tokens(vocab_data), min_freq=2, specials=[UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN])
src_vocab.set_default_index(src_vocab[UNK_TOKEN]) # set the unknown token '<unk>' as default

# Build a vocabulary for English
vocab_data = Multi30k(split=('train'), language_pair=('de', 'en'))
trg_vocab = build_vocab_from_iterator(yield_trg_tokens(vocab_data), min_freq=2, specials=[UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN])
trg_vocab.set_default_index(trg_vocab[UNK_TOKEN]) # set the unknown token '<unk>' as default

# A function to process each batch
def collate_fn(batch):
  src_list, trg_list = [], []
  for src_sentence, trg_sentence in batch:
    src_tokens = src_tokenizer(src_sentence.strip().lower())
    src_indices = src_vocab([SOS_TOKEN] + src_tokens + [EOS_TOKEN])
    src_list.append(torch.tensor(src_indices, dtype=torch.long))

    trg_tokens = trg_tokenizer(trg_sentence.strip().lower())
    trg_indices = trg_vocab([SOS_TOKEN] + trg_tokens + [EOS_TOKEN])
    trg_list.append(torch.tensor(trg_indices, dtype=torch.long))
  
  src_tensor = pad_sequence(src_list, padding_value=src_vocab[PAD_TOKEN])
  trg_tensor = pad_sequence(trg_list, padding_value=trg_vocab[PAD_TOKEN])
  return src_tensor, trg_tensor

train_data, val_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))
train_data = to_map_style_dataset(train_data)
val_data = to_map_style_dataset(val_data)
test_data = to_map_style_dataset(test_data)

def format_time(start_time, current_time, progress):
  elapsed = int(current_time - start_time)
  elapsed_time = f'{elapsed // 60:2d}m {elapsed % 60:2d}s'
  total = int(elapsed / progress)
  total_time = f'{total // 60:2d}m {total % 60:2d}s'
  return elapsed_time, total_time
  
# A function for training
def train(dataloader, model, optimizer, loss_fn, device, tf_ratio=0., verbose=True, print_every=50):
  model.train()
  avg_loss = 0.
  loss_history = []
  model.train()
  for batch, (src, trg) in enumerate(dataloader):
    src, trg = src.to(device), trg.to(device)
    pred, _ = model(src, trg, tf_ratio=tf_ratio)
    pred = pred[1:].view(-1, pred.size(2))
    trg = trg[1:].view(-1)
    # pred: [trg_len * batch_size, output_size], trg: [trg_len * batch_size]
    loss = loss_fn(pred, trg)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_loss += loss.item()
    loss_history.append(loss.item())
    if verbose and batch % print_every == 0 and batch > 0:
      avg_loss /= print_every
      print(f'> [{(batch + 1) * src.size(1):5d}/{len(dataloader.dataset):5d}]',
            f'loss={avg_loss:1.4f}, ppl={exp(avg_loss):7.3f}')
      avg_loss = 0.
    
  return loss_history

# A function for evaluation
def evaluate(dataloader, model, loss_fn, device, verbose=True):
  avg_loss = 0.
  loss_history = []
  model.eval()
  with torch.no_grad():
    for batch, (src, trg) in enumerate(dataloader):
      src, trg = src.to(device), trg.to(device)
      pred, _ = model(src, trg)
      pred = pred[1:trg.size(0)].view(-1, pred.size(2))
      trg = trg[1:].view(-1)
      # pred: [trg_len * batch_size, output_size], trg: [trg_len * batch_size]
      loss = loss_fn(pred, trg)

      avg_loss += loss.item()
      loss_history.append(loss.item())
  if verbose:
    avg_loss /= len(dataloader)
    print(f'> [evaluation]  loss={avg_loss:1.4f},',
          f'ppl={exp(avg_loss):7.3f}')
  return avg_loss, loss_history
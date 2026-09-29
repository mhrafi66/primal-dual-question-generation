import torch
import torch.nn as nn
from transformers import AutoModel
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import math
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
torch.manual_seed(66)

def _env_int(name, default):
  return int(os.environ.get(name, default))

def _env_float(name, default):
  return float(os.environ.get(name, default))

def _env_bool(name, default=False):
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in {"1", "true", "yes", "on"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Lightweight experiment controls. Defaults preserve the restored class-submission
# behavior; environment variables make smoke tests and ablations reproducible.
PDQG_MAX_TRAIN_SAMPLES = _env_int("PDQG_MAX_TRAIN_SAMPLES", 0)
PDQG_MAX_EVAL_SAMPLES = _env_int("PDQG_MAX_EVAL_SAMPLES", 0)
PDQG_RUN_NAME = os.environ.get("PDQG_RUN_NAME", "full")

train_dataset = load_dataset("squad", split="train")
test_dataset = load_dataset("squad", split="validation")

if PDQG_MAX_TRAIN_SAMPLES > 0:
  train_dataset = train_dataset.select(range(min(PDQG_MAX_TRAIN_SAMPLES, len(train_dataset))))
if PDQG_MAX_EVAL_SAMPLES > 0:
  test_dataset = test_dataset.select(range(min(PDQG_MAX_EVAL_SAMPLES, len(test_dataset))))

model_name = 'facebook/bart-base'

#Data Tokenize and Preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)

def find_answer_token_span(offset_mapping, answer_start_char, answer_end_char):
  """Map a character-level SQuAD answer span to token indices.

  Returns (-100, -100) if truncation removed the answer. CrossEntropyLoss
  uses -100 as an ignored label.
  """
  token_start = None
  token_end = None

  for idx, (start, end) in enumerate(offset_mapping):
    if start == end:  # special tokens such as <s> and </s>
      continue
    if token_start is None and start <= answer_start_char < end:
      token_start = idx
    if start < answer_end_char <= end:
      token_end = idx
      break

  if token_start is None or token_end is None:
    return -100, -100
  return token_start, token_end

def tokenize_single_sample(dict_row):
  context = dict_row['context'].strip()
  context_tokens = tokenizer.tokenize(context, truncation=True, max_length=512)

  question = dict_row['question'].strip()
  question_tokens = tokenizer.tokenize(question, truncation=True, max_length=512)
  question_input_ids = tokenizer(question, truncation=True, max_length=512)['input_ids']
  quesiton_input_positions = torch.tensor(list(range(len(question_input_ids))))

  answer_text = dict_row['answers']['text'][0].strip()
  answer_tokens = tokenizer.tokenize(answer_text, truncation=True, max_length=512)
  answer_start_char = dict_row['answers']['answer_start'][0]
  answer_end_char = answer_start_char + len(answer_text)


  # Section-1: Question Generation (Input: will be Context and Answer text, Output: Question)
  tokenized_input_qg = tokenizer(context +'</s><s>'+ answer_text, truncation=True, max_length=512)['input_ids']
  # task=0 for Question Generation, 1 for Question Answering and 2 for Uncommon Word generation
  task = 0
  tokenized_input_position_qg = torch.tensor(list(range(len(context_tokens)+2)) + list(range(len(answer_tokens)+2)))
  task_embedding_input_qg = torch.tensor([task] * len(tokenized_input_qg))
  # Segment ID = 0 for context, 1 for answer and 2 for question. and +2 is for token <s> and </s> for each segment
  segment_embedding_input_qg = torch.tensor([0] * (len(context_tokens)+2) + [1] * (len(answer_tokens)+2))
  #The following code I am writing for just patching in error handling
  if segment_embedding_input_qg.size(0) > 512:
    desired_length_a = int(0.95 * len(tokenized_input_qg))
    desired_length_b = len(tokenized_input_qg) - desired_length_a
    segment_embedding_input_qg = torch.tensor([0] * desired_length_a  + [1] * desired_length_b)

  # Section-2: Question Answering (corrected primal-dual path)
  # The original submission concatenated context + *ground-truth question* here.
  # That leaked the reference question into the dual QA task and disconnected QA
  # loss from the generated-question representation. The corrected model embeds
  # only the passage here; generated question embeddings are appended in forward().
  qa_question_budget = max(1, len(question_input_ids) - 1)
  qa_context_max_length = max(8, 512 - qa_question_budget)
  qa_context_encoding = tokenizer(
      context,
      truncation=True,
      max_length=qa_context_max_length,
      return_offsets_mapping=True
  )
  tokenized_input_qa = qa_context_encoding['input_ids']
  answer_token_start, answer_token_end = find_answer_token_span(
      qa_context_encoding['offset_mapping'],
      answer_start_char,
      answer_end_char
  )
  tokenized_input_position_qa = torch.tensor(list(range(len(tokenized_input_qa))))
  task_embedding_input_qa = torch.tensor([1] * len(tokenized_input_qa))
  segment_embedding_input_qa = torch.tensor([0] * len(tokenized_input_qa))

  # Section-3: Uncommon Word Generation (Input: Context, Output: Word Distribution)
  tokenized_input_uw = tokenizer(context, truncation=True, max_length=512)['input_ids']
  # task=0 for Question Generation, 1 for Question Answering and 2 for Uncommon Word generation
  task = 2
  tokenized_input_position_uw = torch.tensor(list(range(len(context_tokens)+2)))
  task_embedding_input_uw = torch.tensor([task] * len(tokenized_input_uw))
  # Segment ID = 0 for context, 1 for answer and 2 for question. and +2 is for token <s> and </s> for each segment
  segment_embedding_input_uw = torch.tensor([0] * (len(context_tokens)+2))
  if segment_embedding_input_uw.size(0) > 512:
    segment_embedding_input_uw = torch.tensor([0] * len(tokenized_input_uw))

  return {
      'context_tokens': context_tokens,
      'question_tokens': question_tokens,
      'question_input_ids': question_input_ids,
      'quesiton_input_positions': quesiton_input_positions,
      'answer_tokens': answer_tokens,
      'answer_token_start': answer_token_start,
      'answer_token_end': answer_token_end,
      'tokenized_input_qg': tokenized_input_qg,
      'tokenized_input_position_qg': tokenized_input_position_qg,
      'task_embedding_input_qg': task_embedding_input_qg,
      'segment_embedding_input_qg': segment_embedding_input_qg,
      'tokenized_input_qa': tokenized_input_qa,
      'tokenized_input_position_qa': tokenized_input_position_qa,
      'task_embedding_input_qa': task_embedding_input_qa,
      'segment_embedding_input_qa': segment_embedding_input_qa,
      'tokenized_input_uw': tokenized_input_uw,
      'tokenized_input_position_uw': tokenized_input_position_uw,
      'task_embedding_input_uw': task_embedding_input_uw,
      'segment_embedding_input_uw': segment_embedding_input_uw
  }


train_dataset = train_dataset.map(tokenize_single_sample)
test_dataset = test_dataset.map(tokenize_single_sample)

#Model Definition
class EmbeddingLayer(nn.Module):
  def __init__(self, embedding_dim):
    super(EmbeddingLayer, self).__init__()
    self.embedding_dim = embedding_dim
    self.bart_embedding = AutoModel.from_pretrained(model_name).shared
    # #Converting BART embedding to desired dimension
    # self.word_embedding = nn.Linear(self.bart_embedding.weight.shape[1], embedding_dim)
    #position embedding is done combined with word embedding in bart
    # self.positional_embedding = AutoModel.from_pretrained(model_name).get_input_embeddings().position_embeddings
    self.task_embedding = nn.Embedding(3, embedding_dim)
    self.segment_embedding = nn.Embedding(3, embedding_dim)


  def forward(self, word_input, position_input, task_input, segment_input):
    # print(word_input.shape)
    word_input = self.bart_embedding(word_input)
    # # print(word_input.shape)
    # word_input = self.word_embedding(word_input)
    # print(word_input.shape)
    # position_input = self.positional_embedding(position_input)
    task_input = self.task_embedding(task_input)
    # print(task_input.shape)
    segment_input = self.segment_embedding(segment_input)
    # print(segment_input.shape)

    embedding_output = word_input + (task_input + segment_input) / math.sqrt(self.embedding_dim)
    return embedding_output

def create_distillation_mask(input_tokens):
  num_rows, num_cols = input_tokens.shape
  percent_true = 0.10
  num_cols -= 2 #subtracting 2 as input id contains <s> and </s>
  # Number of True elements per row
  num_true_per_row = int(num_cols * percent_true)

  # Create and shuffle each row, then stack them into a 2D tensor
  rows = [torch.cat((torch.ones(num_true_per_row, dtype=torch.bool),
                    torch.zeros(num_cols - num_true_per_row, dtype=torch.bool)))[torch.randperm(num_cols)]
          for _ in range(num_rows)]
  b = torch.stack(rows)

  # Add False columns at the beginning and end of each row for <s> and </s>
  b = torch.cat((torch.zeros(num_rows, 1, dtype=torch.bool), b, torch.zeros(num_rows, 1, dtype=torch.bool)), dim=1)

  return b


class QuesitonGenerationWithKnowledgeDist(nn.Module):
  def __init__(self, embedding_dim, vocab_size, pad_token_id):
    super(QuesitonGenerationWithKnowledgeDist, self).__init__()
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.pad_token_id = pad_token_id

    self.embedding_layer = EmbeddingLayer(embedding_dim = embedding_dim)
    self.primal_dual_encoder = AutoModel.from_pretrained(model_name).encoder
    self.question_decoder = AutoModel.from_pretrained(model_name).decoder
    self.output_layer_qg = AutoModelForSeq2SeqLM.from_pretrained(model_name).lm_head
    self.start_index_ff_qa = nn.Linear(self.embedding_dim, 1)
    self.end_index_ff_qa = nn.Linear(self.embedding_dim*2, 1)
    self.softmax = nn.Softmax(dim = -1)

    self.pretrained_model_no_grad_uw = AutoModel.from_pretrained(model_name)
    self.Wm_uw = nn.Linear(self.embedding_dim, self.vocab_size)



  def forward(self, tokenized_input_qg, tokenized_input_position_qg, task_embedding_input_qg, segment_embedding_input_qg, question_input_ids,
              tokenized_input_qa, tokenized_input_position_qa, task_embedding_input_qa, segment_embedding_input_qa,
              tokenized_input_uw, tokenized_input_position_uw, task_embedding_input_uw, segment_embedding_input_uw):
    #Question Generation Task
    qg_embedded = self.embedding_layer(tokenized_input_qg, tokenized_input_position_qg, task_embedding_input_qg, segment_embedding_input_qg)
    qg_attention_mask = tokenized_input_qg != self.pad_token_id
    # qg_attention_mask = torch.ones_like(tokenized_input_qg)
    # qg_attention_mask[tokenized_input_qg == self.pad_token_id] = 0
    # print(qg_attention_mask.shape)
    qg_encoded_last_hidden_state = self.primal_dual_encoder(inputs_embeds=qg_embedded, attention_mask=qg_attention_mask).last_hidden_state
    # print("Last Layer of encoder =",qg_encoded_last_hidden_state.shape)
    # qg_question_id_attention_mask = torch.ones_like(question_input_ids)
    # # print("Question Input =",question_input_ids.shape)
    # qg_question_id_attention_mask[question_input_ids == self.pad_token_id] =
    trimmed_question_input_ids = question_input_ids[:, :-1]
    qg_question_id_attention_mask = trimmed_question_input_ids != self.pad_token_id
    qg_decoded_last_hidden_state = self.question_decoder(input_ids=trimmed_question_input_ids, attention_mask=qg_question_id_attention_mask, encoder_hidden_states=qg_encoded_last_hidden_state).last_hidden_state
    # print("Last Layer of decoder =",qg_decoded_last_hidden_state.shape)
    qg_generated_questions = self.output_layer_qg(qg_decoded_last_hidden_state)
    # print("Feed Forward Layer Output =",qg_generated_questions.shape)



    # Question Answering Task -- corrected primal-dual coupling
    # Passage embeddings come from the QA input. The question representation is
    # the decoder hidden state produced by the QG branch above, NOT an embedding
    # of the ground-truth question. This makes QA loss train the shared QG path.
    qa_context_embedded = self.embedding_layer(
        tokenized_input_qa, tokenized_input_position_qa,
        task_embedding_input_qa, segment_embedding_input_qa
    )
    qa_context_attention_mask = tokenized_input_qa != self.pad_token_id

    generated_question_task = torch.ones_like(trimmed_question_input_ids)
    generated_question_segment = torch.full_like(trimmed_question_input_ids, 2)
    generated_question_qa = qg_decoded_last_hidden_state + (
        self.embedding_layer.task_embedding(generated_question_task)
        + self.embedding_layer.segment_embedding(generated_question_segment)
    ) / math.sqrt(self.embedding_dim)

    qa_embedded = torch.cat((qa_context_embedded, generated_question_qa), dim=1)
    qa_attention_mask = torch.cat(
        (qa_context_attention_mask, qg_question_id_attention_mask), dim=1
    )

    qa_encoded_last_hidden_state = self.primal_dual_encoder(
        inputs_embeds=qa_embedded, attention_mask=qa_attention_mask
    ).last_hidden_state

    # The answer must come from the passage, never from the appended question.
    context_answer_mask = qa_context_attention_mask.clone()
    context_answer_mask &= tokenized_input_qa != tokenizer.bos_token_id
    context_answer_mask &= tokenized_input_qa != tokenizer.eos_token_id
    answer_candidate_mask = torch.cat(
        (context_answer_mask, torch.zeros_like(qg_question_id_attention_mask)),
        dim=1
    )

    start_index_logits = self.start_index_ff_qa(qa_encoded_last_hidden_state).squeeze(-1)
    start_index_logits = start_index_logits.masked_fill(~answer_candidate_mask, float('-inf'))
    start_index_sm = self.softmax(start_index_logits)
    final_start_idx = torch.argmax(start_index_sm, dim=-1, keepdim=True)

    positions = torch.arange(qa_attention_mask.size(1), device=qa_attention_mask.device)
    # End may equal start for one-token answers; only positions before start are invalid.
    ignore_for_end_index_mask = positions < final_start_idx
    ignore_for_end_index_mask = ignore_for_end_index_mask | ~answer_candidate_mask

    repeated_start_state = torch.gather(
        qa_encoded_last_hidden_state,
        1,
        final_start_idx.unsqueeze(2).expand(
            -1, qa_encoded_last_hidden_state.size(1), qa_encoded_last_hidden_state.size(2)
        )
    )
    qa_end_features = torch.cat(
        (qa_encoded_last_hidden_state, repeated_start_state), dim=-1
    )
    end_index_logits = self.end_index_ff_qa(qa_end_features).squeeze(-1)
    end_index_logits = end_index_logits.masked_fill(
        ignore_for_end_index_mask, float('-inf')
    )
    end_index_sm = self.softmax(end_index_logits)



    #Uncommon Word Generation
    uw_embedded = self.embedding_layer(tokenized_input_uw, tokenized_input_position_uw, task_embedding_input_uw, segment_embedding_input_uw)
    uw_attention_mask = tokenized_input_uw != self.pad_token_id
    # qa_attention_mask[tokenized_input_qa == self.pad_token_id] = 0
    uw_encoded_last_hidden_state = self.primal_dual_encoder(inputs_embeds=uw_embedded, attention_mask=uw_attention_mask).last_hidden_state
    uw_distillation_mask = create_distillation_mask(tokenized_input_uw).to(device)
    with torch.no_grad():
      uw_pretrained_encoded_last_hidden_state = self.pretrained_model_no_grad_uw(input_ids=tokenized_input_uw, attention_mask=uw_distillation_mask, decoder_input_ids = tokenized_input_uw).last_hidden_state
    masked_word_encoding_pre = uw_pretrained_encoded_last_hidden_state.masked_select(uw_distillation_mask.unsqueeze(-1)).view(-1, uw_pretrained_encoded_last_hidden_state.shape[-1])
    y_pre = self.softmax(self.Wm_uw(masked_word_encoding_pre))
    # print(masked_word_encoding_pre.shape)
    masked_word_encoding_en = uw_encoded_last_hidden_state.masked_select(uw_distillation_mask.unsqueeze(-1)).view(-1, uw_encoded_last_hidden_state.shape[-1])
    y_en = self.softmax(self.Wm_uw(masked_word_encoding_en))



    return qg_generated_questions, start_index_logits, end_index_logits, start_index_sm, end_index_sm, y_pre, y_en
  
#Training_Model
##Custom Collate Function
def padding_all_tokens(all_tokens):
    max_length = max(len(token_row) for token_row in all_tokens)
    padded_tokens = []
    for token in all_tokens:
        padded_row = token + ['<pad>'] * (max_length - len(token))
        padded_tokens.append(padded_row)
    return padded_tokens

def padding_all_sequences(all_sequences):
    return pad_sequence(all_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

def custom_collate(batch):
  context_tokens = padding_all_tokens([item['context_tokens'] for item in batch])

  question_tokens = padding_all_tokens([item['question_tokens'] for item in batch])
  question_input_ids = padding_all_sequences([torch.tensor(item['question_input_ids']) for item in batch])
  quesiton_input_positions = padding_all_sequences([torch.tensor(item['quesiton_input_positions']) for item in batch])

  answer_tokens = padding_all_tokens([item['answer_tokens'] for item in batch])
  answer_token_start = torch.tensor([torch.tensor(item['answer_token_start']) for item in batch])
  answer_token_end = torch.tensor([torch.tensor(item['answer_token_end']) for item in batch])

  tokenized_input_qg = padding_all_sequences([torch.tensor(item['tokenized_input_qg']) for item in batch])
  tokenized_input_position_qg = padding_all_sequences([torch.tensor(item['tokenized_input_position_qg']) for item in batch])
  task_embedding_input_qg = padding_all_sequences([torch.tensor(item['task_embedding_input_qg']) for item in batch])
  segment_embedding_input_qg = padding_all_sequences([torch.tensor(item['segment_embedding_input_qg']) for item in batch])

  tokenized_input_qa = padding_all_sequences([torch.tensor(item['tokenized_input_qa']) for item in batch])
  tokenized_input_position_qa = padding_all_sequences([torch.tensor(item['tokenized_input_position_qa']) for item in batch])
  task_embedding_input_qa = padding_all_sequences([torch.tensor(item['task_embedding_input_qa']) for item in batch])
  segment_embedding_input_qa = padding_all_sequences([torch.tensor(item['segment_embedding_input_qa']) for item in batch])

  tokenized_input_uw = padding_all_sequences([torch.tensor(item['tokenized_input_uw']) for item in batch])
  tokenized_input_position_uw = padding_all_sequences([torch.tensor(item['tokenized_input_position_uw']) for item in batch])
  task_embedding_input_uw = padding_all_sequences([torch.tensor(item['task_embedding_input_uw']) for item in batch])
  segment_embedding_input_uw = padding_all_sequences([torch.tensor(item['segment_embedding_input_uw']) for item in batch])

  return {
      'context_tokens': context_tokens,
      'question_tokens': question_tokens,
      'question_input_ids': question_input_ids,
      'quesiton_input_positions': quesiton_input_positions,
      'answer_tokens': answer_tokens,
      'answer_token_start': answer_token_start,
      'answer_token_end': answer_token_end,
      'tokenized_input_qg': tokenized_input_qg,
      'tokenized_input_position_qg': tokenized_input_position_qg,
      'task_embedding_input_qg': task_embedding_input_qg,
      'segment_embedding_input_qg': segment_embedding_input_qg,
      'tokenized_input_qa': tokenized_input_qa,
      'tokenized_input_position_qa': tokenized_input_position_qa,
      'task_embedding_input_qa': task_embedding_input_qa,
      'segment_embedding_input_qa': segment_embedding_input_qa,
      'tokenized_input_uw': tokenized_input_uw,
      'tokenized_input_position_uw': tokenized_input_position_uw,
      'task_embedding_input_uw': task_embedding_input_uw,
      'segment_embedding_input_uw': segment_embedding_input_uw
  }

#Training_Phase
model_name = 'facebook/bart-base'

dimension_of_model = AutoModel.from_pretrained(model_name).shared.weight.shape[1]
print(dimension_of_model)

batch_size = _env_int("PDQG_BATCH_SIZE", 32)
num_epochs = _env_int("PDQG_EPOCHS", 9)
lr = _env_float("PDQG_LR", 3e-5)

alpha = _env_float("PDQG_ALPHA", 0.8)
beta = _env_float("PDQG_BETA", 0.15)
shuffle_train = _env_bool("PDQG_SHUFFLE", False)
checkpoint_path = os.environ.get(
    "PDQG_CHECKPOINT", f"checkpoints/{PDQG_RUN_NAME}.pth"
)

# store_train_dataset = train_dataset
# store_test_dataset = test_dataset

# indices = list(range(20))
# train_dataset = store_train_dataset.select(indices)
# indices = list(range(10))
# test_dataset = store_test_dataset.select(indices)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn= custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn= custom_collate)

model = QuesitonGenerationWithKnowledgeDist(embedding_dim= dimension_of_model, vocab_size = tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
qg_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
qa_criterion = nn.CrossEntropyLoss(ignore_index=-100)

total_training_steps = len(train_dataloader) * num_epochs
learning_rate_warmup_steps = 50
scheduler = get_linear_schedule_with_warmup(optimizer, learning_rate_warmup_steps, total_training_steps)

best_val_loss = float('inf')
best_model = None
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('artifacts/experiments', exist_ok=True)
with open(f'artifacts/experiments/{PDQG_RUN_NAME}_config.json', 'w') as config_file:
  json.dump({
      "run_name": PDQG_RUN_NAME,
      "device": str(device),
      "model": model_name,
      "batch_size": batch_size,
      "epochs": num_epochs,
      "learning_rate": lr,
      "alpha": alpha,
      "beta": beta,
      "max_train_samples": PDQG_MAX_TRAIN_SAMPLES,
      "max_eval_samples": PDQG_MAX_EVAL_SAMPLES,
      "checkpoint": checkpoint_path,
  }, config_file, indent=2)
print("experiment config:", {
    "run": PDQG_RUN_NAME, "alpha": alpha, "beta": beta,
    "epochs": num_epochs, "batch_size": batch_size,
    "max_train": PDQG_MAX_TRAIN_SAMPLES, "max_eval": PDQG_MAX_EVAL_SAMPLES
})

for epoch in range(num_epochs):
  print(f"Epoch {epoch + 1}/{num_epochs}")
  model.train()
  train_loss = 0
  for batch in tqdm(train_dataloader):
    optimizer.zero_grad()
    gen_qs, start_logits, end_logits, start_smax, end_smax, y_pre, y_en = model(batch['tokenized_input_qg'].to(device), batch['tokenized_input_position_qg'].to(device), batch['task_embedding_input_qg'].to(device), batch['segment_embedding_input_qg'].to(device), batch['question_input_ids'].to(device),
                                                                                batch['tokenized_input_qa'].to(device), batch['tokenized_input_position_qa'].to(device), batch['task_embedding_input_qa'].to(device), batch['segment_embedding_input_qa'].to(device),
                                                                                batch['tokenized_input_uw'].to(device), batch['tokenized_input_position_uw'].to(device), batch['task_embedding_input_uw'].to(device), batch['segment_embedding_input_uw'].to(device))
    #Trimming the input ID shape by one to match the shape of input
    #During input, its tail was dropped to save it from producing <\s> by bias
    target_qs = batch['question_input_ids'][:, 1:].to(device)
    qg_loss = qg_criterion(gen_qs.view(-1, gen_qs.shape[-1]), target_qs.view(-1))

    answer_start_pos = batch['answer_token_start'].to(device)
    answer_end_pos = batch['answer_token_end'].to(device)
    # print(criterion(start_logits, answer_start_pos).item(), criterion(end_logits, answer_end_pos).item())
    # print("Start Logits = ",start_logits)
    # print("Start Positions = ",answer_start_pos)
    # print("End Logits = ",end_logits)
    # print("End Positions = ",answer_end_pos)
    qa_loss = qa_criterion(start_logits, answer_start_pos) + qa_criterion(end_logits, answer_end_pos)
    # print(criterion(start_logits, answer_start_pos).item(), criterion(end_logits, answer_end_pos).item())
    uw_loss = -torch.sum(y_en * torch.log(y_pre))
    # print(qg_loss.item(), qa_loss.item(), uw_loss.item())
    loss_in_epoch = qg_loss + alpha * qa_loss + beta * uw_loss
    loss_in_epoch.backward()
    optimizer.step()
    scheduler.step()
    # print(loss_in_epoch.item())
    train_loss += loss_in_epoch.item()
  print("Train Loss = ",train_loss/len(train_dataloader))

  model.eval()
  val_loss = 0
  with torch.no_grad():
    for batch in tqdm(test_dataloader):
      gen_qs, start_logits, end_logits, start_smax, end_smax, y_pre, y_en = model(batch['tokenized_input_qg'].to(device), batch['tokenized_input_position_qg'].to(device), batch['task_embedding_input_qg'].to(device), batch['segment_embedding_input_qg'].to(device), batch['question_input_ids'].to(device),
                                                                                batch['tokenized_input_qa'].to(device), batch['tokenized_input_position_qa'].to(device), batch['task_embedding_input_qa'].to(device), batch['segment_embedding_input_qa'].to(device),
                                                                                batch['tokenized_input_uw'].to(device), batch['tokenized_input_position_uw'].to(device), batch['task_embedding_input_uw'].to(device), batch['segment_embedding_input_uw'].to(device))
      target_qs = batch['question_input_ids'][:, 1:].to(device)
      qg_loss = qg_criterion(gen_qs.view(-1, gen_qs.shape[-1]), target_qs.view(-1))

      answer_start_pos = batch['answer_token_start'].to(device)
      answer_end_pos = batch['answer_token_end'].to(device)
      qa_loss = qa_criterion(start_logits, answer_start_pos) + qa_criterion(end_logits, answer_end_pos)

      uw_loss = -torch.sum(y_en * torch.log(y_pre))

      loss_in_epoch = qg_loss + alpha * qa_loss + beta * uw_loss
      # print(loss_in_epoch.item())
      val_loss += loss_in_epoch.item()
  val_loss = val_loss/len(test_dataloader)
  print("Validation Loss = ",val_loss)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        "model_param": model.state_dict(),
        "optim_param": optimizer.state_dict(),
        "bst_dev_loss": best_val_loss,
        "epoch": epoch,
        "learning_rate": lr},
               checkpoint_path)


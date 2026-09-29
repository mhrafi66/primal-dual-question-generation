import torch
import torch.nn as nn
from transformers import AutoModel
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
torch.manual_seed(66)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset = load_dataset("squad", split="train")
test_dataset = load_dataset("squad", split="validation")

model_name = 'facebook/bart-base'

#Data Tokenize and Preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)

def tokenize_single_sample(dict_row):
  context = dict_row['context'].strip()
  context_tokens = tokenizer.tokenize(context, truncation=True, max_length=512)

  question = dict_row['question'].strip()
  question_tokens = tokenizer.tokenize(question, truncation=True, max_length=512)
  question_input_ids = tokenizer(question, truncation=True, max_length=512)['input_ids']
  quesiton_input_positions = torch.tensor(list(range(len(question_input_ids))))

  answer_text = dict_row['answers']['text'][0].strip()
  answer_tokens = tokenizer.tokenize(answer_text, truncation=True, max_length=512)
  answer_start = dict_row['answers']['answer_start'][0]
  answer_end = answer_start + len(answer_text) - 1
  # answer token start and end position at the context token array
  answer_token_start = len(tokenizer(context[:answer_start], truncation=True, max_length=512)['input_ids']) - 1
  answer_token_end = len(tokenizer(context[:answer_end], truncation=True, max_length=512)['input_ids']) - 1


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

  # Section-2: Answering Questions (Input: Context and Question, Output: Answer )
  tokenized_input_qa = tokenizer(context + '</s><s>' +question, truncation=True, max_length=512)['input_ids']
  # task=0 for Question Generation, 1 for Question Answering and 2 for Uncommon Word generation
  task = 1
  tokenized_input_position_qa = torch.tensor(list(range(len(context_tokens)+2)) + list(range(len(question_tokens)+2)))
  task_embedding_input_qa = torch.tensor([task] * len(tokenized_input_qa))
  # Segment ID = 0 for context, 1 for answer and 2 for question. and +2 is for token <s> and </s> for each segment
  segment_embedding_input_qa = torch.tensor([0] * (len(context_tokens)+2) + [2] * (len(question_tokens)+2))
  #The following code I am writing for just patching in error handling
  if segment_embedding_input_qa.size(0) > 512:
    desired_length_a = int(0.90 * len(tokenized_input_qa))
    desired_length_b = len(tokenized_input_qa) - desired_length_a
    segment_embedding_input_qa = torch.tensor([0] * desired_length_a  + [1] * desired_length_b)

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



    #Question Answering Task
    qa_embedded = self.embedding_layer(tokenized_input_qa, tokenized_input_position_qa, task_embedding_input_qa, segment_embedding_input_qa)
    qa_attention_mask = tokenized_input_qa != self.pad_token_id
    # qa_attention_mask[tokenized_input_qa == self.pad_token_id] = 0
    qa_encoded_last_hidden_state = self.primal_dual_encoder(inputs_embeds=qa_embedded, attention_mask=qa_attention_mask).last_hidden_state
    # print(qa_encoded_last_hidden_state.shape)
    start_index_logits = self.start_index_ff_qa(qa_encoded_last_hidden_state).squeeze(-1)
    # qa_attention_mask = qa_attention_mask.bool()
    # start_index_logits[~qa_attention_mask] = float('-inf')
    start_index_logits_masked = start_index_logits.masked_fill(~qa_attention_mask, float('-inf'))
    start_index_sm = self.softmax(start_index_logits_masked)
    final_start_idx = torch.argmax(start_index_sm, dim=-1, keepdim=True)

    # print(final_start_idx.shape)
    # ignore_for_end_index_mask = (torch.arange(qa_attention_mask.size(1)) <= final_start_idx)  | ~qa_attention_mask
    temp_tensor = torch.arange(qa_attention_mask.size(1))
    temp_tensor = temp_tensor.to(device)
    ignore_for_end_index_mask = temp_tensor <= final_start_idx
    ignore_for_end_index_mask = ignore_for_end_index_mask | ~qa_attention_mask
    repeated_tensor_of_final_start_index = torch.gather(qa_encoded_last_hidden_state, 1, final_start_idx.unsqueeze(2).expand(-1, qa_encoded_last_hidden_state.size(1), qa_encoded_last_hidden_state.size(2)))
    qa_encoded_last_hidden_state = torch.cat((qa_encoded_last_hidden_state, repeated_tensor_of_final_start_index), dim=-1)
    end_index_logits = self.end_index_ff_qa(qa_encoded_last_hidden_state).squeeze(-1)
    # end_index_logits[ignore_for_end_index_mask] = float('-inf')
    end_index_logits_masked = end_index_logits.masked_fill(ignore_for_end_index_mask, float('-inf'))
    end_index_sm = self.softmax(end_index_logits_masked)



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

batch_size = 32
num_epochs = 9
lr = 3e-5

alpha = 0.8
beta = 0.15

# store_train_dataset = train_dataset
# store_test_dataset = test_dataset

# indices = list(range(20))
# train_dataset = store_train_dataset.select(indices)
# indices = list(range(10))
# test_dataset = store_test_dataset.select(indices)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn= custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn= custom_collate)

model = QuesitonGenerationWithKnowledgeDist(embedding_dim= dimension_of_model, vocab_size = tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

total_training_steps = len(train_dataloader) * num_epochs
learning_rate_warmup_steps = 50
scheduler = get_linear_schedule_with_warmup(optimizer, learning_rate_warmup_steps, total_training_steps)

best_val_loss = float('inf')
best_model = None

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
    qg_loss = criterion(gen_qs.view(-1, gen_qs.shape[-1]), target_qs.view(-1))

    answer_start_pos = batch['answer_token_start'].to(device)
    answer_end_pos = batch['answer_token_end'].to(device)
    # print(criterion(start_logits, answer_start_pos).item(), criterion(end_logits, answer_end_pos).item())
    # print("Start Logits = ",start_logits)
    # print("Start Positions = ",answer_start_pos)
    # print("End Logits = ",end_logits)
    # print("End Positions = ",answer_end_pos)
    qa_loss = criterion(start_logits, answer_start_pos) + criterion(end_logits, answer_end_pos)
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
      qg_loss = criterion(gen_qs.view(-1, gen_qs.shape[-1]), target_qs.view(-1))

      answer_start_pos = batch['answer_token_start'].to(device)
      answer_end_pos = batch['answer_token_end'].to(device)
      qa_loss = criterion(start_logits, answer_start_pos) + criterion(end_logits, answer_end_pos)

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
               f"/uufs/chpc.utah.edu/common/home/u1472438/NLP_Project/QwithKD1.pth")


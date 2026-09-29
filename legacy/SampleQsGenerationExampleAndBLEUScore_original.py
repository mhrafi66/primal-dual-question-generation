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
import numpy as np
from sacrebleu import corpus_bleu
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
torch.manual_seed(66)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    #Trimming this otherwise the predicted values seems to produce end of sequence token just after the first word
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

#Working Phase
model_name = 'facebook/bart-base'

dimension_of_model = AutoModel.from_pretrained(model_name).shared.weight.shape[1]
print(dimension_of_model)

batch_size = 4
num_epochs = 1
lr = 3e-5

alpha = 0.8
beta = 0.15

def generate_question(model, temp_dataset, beam_size = 10, max_ques_len = 64):
  model.eval()
  with torch.no_grad():
    sequences = [[[tokenizer.bos_token_id], 0.0]]
    for m in range(max_ques_len):
      temp_sequences = []
      for i in range(len(sequences)):
        sequence = sequences[i][0]
        score = sequences[i][1]
        input_question_ids = [sequence + [tokenizer.eos_token_id]]
        ques, sl, el, ssm, esm, y_pre, y_en = model(torch.tensor(temp_dataset['tokenized_input_qg']).to(device), torch.tensor(temp_dataset['tokenized_input_position_qg']).to(device), torch.tensor(temp_dataset['task_embedding_input_qg']).to(device), torch.tensor(temp_dataset['segment_embedding_input_qg']).to(device), torch.tensor(input_question_ids).to(device),
                                                    torch.tensor(temp_dataset['tokenized_input_qa']).to(device), torch.tensor(temp_dataset['tokenized_input_position_qa']).to(device), torch.tensor(temp_dataset['task_embedding_input_qa']).to(device), torch.tensor(temp_dataset['segment_embedding_input_qa']).to(device),
                                                    torch.tensor(temp_dataset['tokenized_input_uw']).to(device), torch.tensor(temp_dataset['tokenized_input_position_uw']).to(device), torch.tensor(temp_dataset['task_embedding_input_uw']).to(device), torch.tensor(temp_dataset['segment_embedding_input_uw']).to(device))
        last_prediction = ques[0,-1]
        top_k_prob, top_k_ids = torch.topk(torch.softmax(last_prediction, dim = -1), beam_size)
        for j in range(beam_size):
          temp_seq = sequence + [top_k_ids[j].item()]
          temp_score = np.log(top_k_prob[j].item())
          temp_sequences.append([temp_seq,temp_score])
      temp_sequences = sorted(temp_sequences, key=lambda x: x[1], reverse=True)
      # print(temp_sequences)
      sequences = temp_sequences[:beam_size]

      results = [n[0][-1] == tokenizer.eos_token_id for n in sequences]
      if all(results):
        break;
    final_ids = sequences[0]
    # print(final_ids)
    qs = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    bleu_score = corpus_bleu([qs], [temp_dataset['question']])

  return qs, bleu_score

##Taking random samples from the test set and generate for them
n = 100
min_value = 0
max_value = len(test_dataset)
random_indices = [random.randint(min_value, max_value) for _ in range(n)]
random_samples = test_dataset.select(random_indices)

model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/NLP_Project/QwithKD1.pth"
checkpoint = torch.load(model_path)
model = QuesitonGenerationWithKnowledgeDist(embedding_dim= dimension_of_model, vocab_size = tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
model = model.to(device)

for i in range(len(random_samples)):
  gen_qs, bscore = generate_question(model, random_samples[i:i+1])
  print(random_samples[i:i+1]['question'])
  print(gen_qs)
  print(bscore.score)


#Calculate BLEU Score for the test set
total_bleu = 0
for i in range(len(test_dataset)):
  gen_qs, bscore = generate_question(model, test_dataset[i:i+1])
  total_bleu += bscore.score
print("BLEU score of the test set = ",total_bleu/len(test_dataset))
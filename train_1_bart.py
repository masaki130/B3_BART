#!/usr/bin/env python
# coding: utf-8

# BARTモデルの学習
# csj+wiki;all_2.json

# In[ ]:


from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer_1.json", 
                                    bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")


# In[ ]:


from transformers import BartForConditionalGeneration, BartConfig
import json

config_facebook_bart_base = json.load(open("config_facebook_bart_base.json", "r"))

del config_facebook_bart_base['_name_or_path']
del config_facebook_bart_base['task_specific_params']
del config_facebook_bart_base['transformers_version']
config_facebook_bart_base['vocab_size'] = tokenizer.vocab_size
config_facebook_bart_base['bos_token_id'] = tokenizer.bos_token_id
config_facebook_bart_base['forced_bos_token_id'] = tokenizer.bos_token_id
config_facebook_bart_base['eos_token_id'] = tokenizer.eos_token_id
config_facebook_bart_base['forced_eos_token_id'] = tokenizer.eos_token_id
config_facebook_bart_base['pad_token_id'] = tokenizer.pad_token_id
config_facebook_bart_base['decoder_start_token_id'] = tokenizer.eos_token_id

config = BartConfig(**config_facebook_bart_base)

model = BartForConditionalGeneration(config)


# In[ ]:


# This function is copied from modeling_bart.py
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


# In[ ]:


def convert_to_features(example_batch):
    # input_encodings = tokenizer.batch_encode_plus(example_batch['tokenized_kana_text'], pad_to_max_length=True, max_length=1024, return_tensors="pt")
    # target_encodings = tokenizer.batch_encode_plus(example_batch['plain_text'], pad_to_max_length=True, max_length=1024, return_tensors="pt")
    
    input_encodings = tokenizer.batch_encode_plus(example_batch['text'], 
                                                  pad_to_max_length=True, max_length=512, 
                                                  # padding=True,
                                                  return_tensors="pt")
    target_encodings = tokenizer.batch_encode_plus(example_batch['phoneme_text'], 
                                                   pad_to_max_length=True, max_length=512, 
                                                   # padding=True,
                                                   return_tensors="pt")

    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id)
    labels[labels[:, :] == tokenizer.pad_token_id] = -100

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }

    return encodings


# In[ ]:


from datasets import Dataset
dataset_all = Dataset.from_json("all_1.json")

# 実際に使うデータセットをここで指定する
# dataset_sub = dataset_all
dataset_sub = dataset_all  # 学習回数を変更5000→
dataset_dict = Dataset.train_test_split(dataset_sub, test_size=0.1)
dataset_dict = dataset_dict.map(convert_to_features, batched=True)

columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',] 
dataset_dict.set_format(type='torch', columns=columns)


# In[ ]:


dataset_dict.save_to_disk("dataset_dict")


# In[ ]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(  
    output_dir='./models/g2p_prosody_bart',
    num_train_epochs=50,
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    weight_decay=0.01,              
    logging_dir='./logs',          
)

# Trainerクラスのインスタンス化
trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=dataset_dict['train'],        
    eval_dataset=dataset_dict['test']
)


# In[ ]:

# bartの学習
trainer.train()     # trainメソッド内を実行、train()はtransformersに定義されている!


# In[ ]:

# データセットの234行目のテキストを入力文に使用
# input_text = dataset_all[234]['text']
# print(input_text)


# In[ ]:


# input_ids=tokenizer.encode(input_text, return_tensors="pt").to("cuda")
# print(input_ids)


# In[ ]:


# output_ids = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True)
# print(output_ids)


# In[ ]:


# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)

# 学習したモデルを保存するパス(trained_bart_modelディレクトリに保存)
output_model_path = "./trained_bart_model"

# 学習済みモデルを保存
trainer.save_model(output_model_path)
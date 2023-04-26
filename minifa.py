# Manifest Utils

import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager

from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import json


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath

VERSION = "cv-corpus-6.1-2020-12-11"
LANGUAGE = "fa"

tokenizer_dir = os.path.join('tokenizers', LANGUAGE)
manifest_dir = os.path.join('manifests', LANGUAGE)

train_manifest = f"{manifest_dir}/commonvoice_train_manifest.json"
dev_manifest = f"{manifest_dir}/commonvoice_dev_manifest.json"
test_manifest = f"{manifest_dir}/commonvoice_test_manifest.json"

train_manifest_data = read_manifest(train_manifest)
dev_manifest_data = read_manifest(dev_manifest)
test_manifest_data = read_manifest(test_manifest)


print('@@@@@@@@',train_manifest_data[0])

train_text = [data['text'] for data in train_manifest_data]
dev_text = [data['text'] for data in dev_manifest_data]
test_text = [data['text'] for data in test_manifest_data]


from collections import defaultdict

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

train_charset = get_charset(train_manifest_data)
dev_charset = get_charset(dev_manifest_data)
test_charset = get_charset(test_manifest_data)

train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
test_set = set(test_charset.keys())


print(f"Number of tokens in train+dev set : {len(train_dev_set)}")
print(f"Number of tokens in test set : {len(test_set)}")


# Preprocessing steps
import re


def contains_only_allowed_chars(str):
    #pattern = r'[^abcd\s]'
    pattern = r'[^آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ\s]'
    matches = re.findall(pattern, str)
    if len(matches) > 0:
        return False
    else:
        return True
        
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\【\】\・\。\『\』\、\ー\〜]'  # remove special character tokens
#kanji_removal_regex = '[' + "".join([f"\{token}" for token in extra_kanji]) + ']'  # remove test set kanji


def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, '', data["text"]).lower().strip()
    
   
    sub='persian_mozilla/'
    
    data["audio_filepath"] = data["audio_filepath"].replace('/content/',sub)

    if contains_only_allowed_chars(data["text"])==False:
      return 0
    else:
      return data



# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    #new_manifest=[]
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            temp = processor(manifest[idx])
            if temp!=0:
              manifest[idx] = temp
            else:
              manifest[idx] = ''

    manifest = [element for element in manifest if element != '']

    print("Finished processing manifest !")
    return manifest


# List of pre-processing functions
PREPROCESSORS = [
    remove_special_characters,
]

# Load manifests
train_data = read_manifest(train_manifest)
dev_data = read_manifest(dev_manifest)
test_data = read_manifest(test_manifest)

# Apply preprocessing
train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
dev_data_processed = apply_preprocessors(dev_data, PREPROCESSORS)
test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

print('@@@@@@@@',train_data_processed[0])

# Write new manifests
train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest)
dev_manifest_cleaned = write_processed_manifest(dev_data_processed, dev_manifest)
test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest)



train_manifest_data = read_manifest(train_manifest_cleaned)
train_charset = get_charset(train_manifest_data)

dev_manifest_data = read_manifest(dev_manifest_cleaned)
dev_charset = get_charset(dev_manifest_data)

train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))


print(f"Number of tokens in preprocessed train+dev set : {len(train_dev_set)}")

###############################



char_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')
char_model.change_vocabulary(new_vocabulary=list(train_dev_set))

for k in list(char_model.cfg.keys()):
  print(k,char_model.cfg[k])
  
#@title Freeze Encoder { display-mode: "form" }
freeze_encoder = True #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)

import torch
import torch.nn as nn

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)
            
if freeze_encoder:
  char_model.encoder.freeze()
  char_model.encoder.apply(enable_bn_se)
  logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
else:
  char_model.encoder.unfreeze()
  logging.info("Model encoder has been un-frozen")
  
char_model.cfg.labels = list(train_dev_set)

cfg = copy.deepcopy(char_model.cfg)

# Setup train, validation, test configs
with open_dict(cfg):    
  # Train dataset  (Concatenate train manifest cleaned and dev manifest cleaned)
  cfg.train_ds.manifest_filepath = f"{train_manifest_cleaned},{dev_manifest_cleaned}"
  cfg.train_ds.labels = list(train_dev_set)
  cfg.train_ds.normalize_transcripts = False
  cfg.train_ds.batch_size = 32
  cfg.train_ds.num_workers = 8
  cfg.train_ds.pin_memory = True
  cfg.train_ds.trim_silence = True

  # Validation dataset  (Use test dataset as validation, since we train using train + dev)
  cfg.validation_ds.manifest_filepath = test_manifest_cleaned
  cfg.validation_ds.labels = list(train_dev_set)
  cfg.validation_ds.normalize_transcripts = False
  cfg.validation_ds.batch_size = 8
  cfg.validation_ds.num_workers = 8
  cfg.validation_ds.pin_memory = True
  cfg.validation_ds.trim_silence = True
  
  
# setup data loaders with new configs
char_model.setup_training_data(cfg.train_ds)
char_model.setup_multiple_validation_data(cfg.validation_ds)


# Original optimizer + scheduler
#print(OmegaConf.to_yaml(char_model.cfg.optim))


with open_dict(char_model.cfg.optim):
  char_model.cfg.optim.lr = 0.01
  char_model.cfg.optim.betas = [0.95, 0.5]  # from paper
  char_model.cfg.optim.weight_decay = 0.001  # Original weight decay
  char_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
  char_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
  char_model.cfg.optim.sched.min_lr = 1e-5
  
# with open_dict(char_model.cfg.spec_augment):
#   char_model.cfg.spec_augment.freq_masks = 2
#   char_model.cfg.spec_augment.freq_width = 25
#   char_model.cfg.spec_augment.time_masks = 2
#   char_model.cfg.spec_augment.time_width = 0.05

char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)
  
#@title Metric
use_cer = True #@param ["False", "True"] {type:"raw"}
log_prediction = True #@param ["False", "True"] {type:"raw"}

char_model._wer.use_cer = use_cer
char_model._wer.log_prediction = log_prediction

import torch
import pytorch_lightning as ptl

if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'cpu'

print('@@@@@@@@@@@@@@@@@@@@@',accelerator)

EPOCHS = 400  # 100 epochs would provide better results, but would take an hour to train

trainer = ptl.Trainer(devices=1, 
                      accelerator=accelerator, 
                      max_epochs=EPOCHS, 
                      accumulate_grad_batches=1,
                      enable_checkpointing=False,
                      logger=False,
                      log_every_n_steps=5,
                      check_val_every_n_epoch=10)

# Setup model with the trainer
char_model.set_trainer(trainer)

# Finally, update the model's internal config
char_model.cfg = char_model._cfg


# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/lang-{LANGUAGE}/',
    name=f"ASR-Char-Model-Language-{LANGUAGE}",
    #resume_if_exists= True,
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)


#%%time
trainer.fit(char_model)

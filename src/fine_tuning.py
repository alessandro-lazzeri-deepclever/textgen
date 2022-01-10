from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer
from transformers import TrainingArguments
from transformers import Trainer
#from datasets import load_dataset
from ovidioDataset import load_dataset
from transformers import AdamW
import torch
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
tokenizer.pad_token = tokenizer.eos_token

train_path = Path('../dataset/ovidio/processed/train.txt')
test_path = Path('../dataset/ovidio/processed/test.txt')

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")

training_args = TrainingArguments(
    output_dir="./geppetto-ovidio", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=500, # number of training epochs
    per_device_train_batch_size=4, # batch size for training
    per_device_eval_batch_size=4,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model()
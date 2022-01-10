from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from transformers import Trainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from pathlib import Path
import json


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #print('inputs',item['input_ids'].shape)
        #print('labels',item['labels'].shape)
        return item

    def __len__(self):
        return len(self.encodings)


def read_recipe(file_path):
    file_path = Path(file_path)
    with open(file_path, 'r') as json_file:
        recipes = json.load(json_file)
    title ='title'
    ingredients ='ingredients'
    instructions = 'instructions'
    texts = []

    for recipe in recipes.items():
        try:
            t = recipe[1][title]+'\n'
            t += ''.join(recipe[1][ingredients]).replace(' ADVERTISEMENT','\n').replace('ADVERTISEMENT','.\n')
            t += ''.join(recipe[1][instructions])
            texts.append(t)
        except KeyError:
            pass
    return texts


if __name__ == '__main__':

    train_file_path = Path('../dataset/recipes_raw/recipes_raw_nosource_ar.json')
    train_texts = read_recipe(train_file_path)

    test_file_path = Path('../dataset/recipes_raw/recipes_raw_nosource_epi.json')
    test_texts = read_recipe(test_file_path)

    train_texts, val_texts = train_test_split(train_texts, test_size=.2)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
#    tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
    tokenizer.pad_token = tokenizer.eos_token
#    tokenizer.model_max_length = 200

    train_text_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_text_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_text_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = RecipeDataset(train_text_encodings)
    val_dataset = RecipeDataset(val_text_encodings)
    test_dataset = RecipeDataset(test_text_encodings)

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    #model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-recipe", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=1000, # number of training epochs
        per_device_train_batch_size=2, # batch size for training
        per_device_eval_batch_size=2,  # batch size for evaluation
        eval_steps = 50, # Number of update steps between two evaluations.
        save_steps= 50, # after # steps model is saved
        warmup_steps= 100,# number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model()

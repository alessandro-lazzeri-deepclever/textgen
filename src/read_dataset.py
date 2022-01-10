from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

if False:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    raw_datasets = load_dataset("imdb")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    print(small_train_dataset)

def loader():
    datasetPath = Path('../dataset/ovidio')

    dataset = []

    for f in datasetPath.iterdir():
        myth = []
        with open(f,'r', encoding='utf8') as oof:
            try:
                for l in oof.readlines():
                    if l != '\n':
                        myth.append(str(l))
            except Exception as e:
                print(oof)
                print(e)
        dataset.append(''.join(myth))
    return dataset


if __name__=='__main__':
    if False:
        dataset = loader()
        print(dataset[0][:100])
        tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
        tokenizer.pad_token = tokenizer.eos_token
        dataset = tokenizer(dataset, padding=True, return_tensors="pt")
        print(tokenizer.decode(dataset["input_ids"][0]))
    
    if True:
        datasetPath = Path('../dataset/ovidio')
        tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
    
        dataset = load_dataset('text', data_files='../dataset/ovidio/temp.txt')


        def tokenize_function(examples):
            return tokenizer(examples["text"])


        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        print(dataset)

    datasetPath = Path('../dataset/ovidio')

    if False:
        for f in datasetPath.iterdir():
            myth = []
            with open(f, 'r', encoding='utf8') as oof:
                try:
                    for l in oof.readlines():
                        if l != '\n':
                            myth.append(str(l))
                except Exception as e:
                    print(oof)
                    print(e)
            with open(Path('../dataset/ovidio/temp.txt'),'w', encoding='utf8') as aaf:
                for r in myth:
                    aaf.writelines(r)
#tokenizer example
if False:
    ex = "A narrare il mutare delle forme in corpi nuovi\n"
    tok_ex = tokenizer(ex)

    print(tok_ex)
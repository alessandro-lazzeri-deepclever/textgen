from transformers import pipeline
from transformers import GPT2Tokenizer

SPLIT = r'(?<=\w\.)\s'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text_generator = pipeline('text-generation', model='../src/gpt2-recipe/checkpoint-1000', tokenizer=tokenizer)
prompts = [
    "Very expensive beef",
    "The fat ass of your mother"]

samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=200,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5
)


for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:",sample_output['generated_text'])
        print()
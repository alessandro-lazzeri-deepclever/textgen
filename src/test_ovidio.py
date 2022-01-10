from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer
import re


SPLIT = r'(?<=\w\.)\s'

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")

text_generator = pipeline('text-generation', model='LorenzoDeMattei/GePpeTto', tokenizer=tokenizer)
prompts = [
    "Il tempo lascia sempre il suo segno: nel cuore e sulla pelle. E, come dicono molti psicologi e soul coach, l’unico modo per essere davvero felici è accettare l’età che avanza, amare se stessi e imparare a lasciare andare (il tempo). Ogni momento della vita ha il suo aspetto evolutivo: la gioventù è spontanea, la maturità è razionale. ",
    "cosa ne pensi dei transessuali?"]

samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=200,
    top_k=50,
    top_p=0.8,
    num_return_sequences=5
)


for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:",sample_output['generated_text'])
        print()
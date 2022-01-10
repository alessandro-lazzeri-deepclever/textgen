from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")

model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-italian-finedtuned-squadv1-it-alfa")
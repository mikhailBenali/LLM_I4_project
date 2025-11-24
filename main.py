from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from datasets import load_dataset
from peft import LoraConfig, TaskType


if __name__ == "__main__":
    try:
        if os.path.exists("./flan-t5-base-model") and os.path.exists("./flan-t5-base-tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained("./flan-t5-base-tokenizer")
            print("Tokenizer loaded from local directory.")
            model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-base-model")
            print("Model loaded from local directory.")
        else:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            print("Tokenizer loaded successfully.")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            print("Model loaded successfully.")

            model.save_pretrained("./flan-t5-base-model")
            tokenizer.save_pretrained("./flan-t5-base-tokenizer")
            print("Model and tokenizer saved locally.")
    except Exception as e:
        print(f"Couldn't load model or tokenizer : {e}")



ds = load_dataset("Helsinki-NLP/opus-100", "en-fr")
sample = ds["train"][0]
input_text = sample["translation"]["en"]
print(f"Input text: {input_text}")
instruction = "Translate English to French: "
input_text = instruction + input_text
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")


peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
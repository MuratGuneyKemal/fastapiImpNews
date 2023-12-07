from fastapi import FastAPI
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from peft import PeftModel, PeftConfig
import uvicorn
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm import LinearSVC
app = FastAPI()
data = {}

@app.on_event('startup')
def init_data():
    print("initializing..")
    if torch.cuda.is_available():
        print("cuda")
    else:
        print("cpu")
    config = PeftConfig.from_pretrained("./Impartial-GenAI")
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, "./Impartial-GenAI").to('cuda' if torch.cuda.is_available() else 'cpu')

    data["model"] = model
    data["tokenizer"] = tokenizer
    return data

@app.get("/")
async def root():
    return {"Is_alive": "true"}

def classify_text(text):
    filename = "model.pickle"
    model = pickle.load(open(filename, "rb"))
    tfidf_file = "tfidf.pickle"
    tfidf = pickle.load(open(tfidf_file, "rb"))
    d = pd.Series([text.lower()])
    df = pd.DataFrame(d)
    test_element = tfidf.transform(df[0])

    pred = model.predict(test_element)
    print(pred)
    if 1 in pred:
        return "right"
    if 0 in pred:
        return "center"
    if 2 in pred:
        return "left"
    else:
        return "Invalid"
    
def make_inference(text, tokenizer, model):
    batch = tokenizer(text, return_token_type_ids=False, return_tensors='pt')
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=500)
    else:
        with torch.cpu.amp.autocast():
            output_tokens = model.generate(**batch, max_new_tokens=500)

    return (tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    
@app.post("/classify")
async def classify(text : str):
    response = classify_text(text)
    return {"Bias": response}

@app.post("/generate")
async def generate(article : str, alignment : str):
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Convert the following {alignment}-biased article to unbiased center format. ### Input: {article} """
    output = make_inference(text, data["tokenizer"], data["model"])
    return {"new_article": output}

if __name__ == '__main__':
    uvicorn.run(f'main:app', host='localhost', port=8000)
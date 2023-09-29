from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_seq_data_input(batch_size):
    return ["The cat sat on the mat."]*batch_size

class InferenceHandler:
    def __init__(self,model):
        self.model = model

    def __call__(self,tokenized_input):
        return self.model(input_ids=tokenized_input["input_ids"],attention_mask=tokenized_input["attention_mask"])

class PreprocessFunction:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    #input: tokenizer, list of input strings
    def __call__(self,text_inputs):
        return self.tokenizer(text_inputs, return_tensors="pt", padding="max_length", max_length=384, truncation=True)

def hf_loader(model_checkpoint,quantize=True):
    DATA_CACHE = './.cache'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3, cache_dir=DATA_CACHE, torchscript=True, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=DATA_CACHE, torchscript=True)
    
    text = ["this is a test string"]*8
    res = tokenizer(text, return_tensors="pt", padding="max_length",truncation=True,max_length=384)
    tokens_tensor = res['input_ids']
    masks_tensors = res['attention_mask']
    dummy_input = [tokens_tensor, masks_tensors]
    
    if quantize:
        quantized_model_int8 = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        inference_model = torch.jit.optimize_for_inference(torch.jit.trace(quantized_model_int8.eval(),dummy_input)).to(device)
    else:
        inference_model = torch.jit.optimize_for_inference(torch.jit.trace(model.eval(),dummy_input)).to(device)
    inference_handler = InferenceHandler(inference_model)
    preprocess_function = PreprocessFunction(tokenizer)
    return inference_handler, preprocess_function

seq_loader_dict = {
    "bert_tiny":lambda : hf_loader("google/bert_uncased_L-2_H-128_A-2"),
    "bert_mini":lambda : hf_loader("google/bert_uncased_L-4_H-256_A-4"),
    "bert_small":lambda : hf_loader("google/bert_uncased_L-4_H-512_A-8"),
    "bert_medium":lambda : hf_loader("google/bert_uncased_L-8_H-512_A-8"),
    "bert_base":lambda : hf_loader("ishan/bert-base-uncased-mnli"),
}


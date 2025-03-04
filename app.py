import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
import torch

class InferlessPythonModel:
    def initialize(self):
        snapshot_download(repo_id='intfloat/multilingual-e5-large',allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to("cuda")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def infer(self, inputs):
        sentence = inputs["sentence"]
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return {"embedding":sentence_embeddings.tolist()[0]}

    def finalize(self, args):
        self.model = None
        self.tokenizer = None

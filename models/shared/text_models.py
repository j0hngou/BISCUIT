import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from abc import ABC, abstractmethod
from open_clip import create_model_from_pretrained, get_tokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @abstractmethod
    def forward(self, sentences, tokenized=True):
        """
        [Batch, Sequence, Hidden] -> [Batch, Hidden]
        """
        pass

class SentenceTransformer(TextEncoder):
    """
    Sentence Transformer
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(SentenceTransformer, self).__init__(model_name)

    def forward(self, inp, tokenized=True):
        if tokenized:
            encoded_input = {k:v.squeeze() for k,v in inp.items()}
            if encoded_input['input_ids'].dim() == 1:
                encoded_input['input_ids'] = encoded_input['input_ids'].unsqueeze(0)
                encoded_input['token_type_ids'] = encoded_input['token_type_ids'].unsqueeze(0)
                encoded_input['attention_mask'] = encoded_input['attention_mask'].unsqueeze(0)
        else:
            encoded_input = self.tokenizer(inp, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

class SigLIP(nn.Module):
    """
    SigLIP
    """
    def __init__(self, model_name='hf-hub:timm/ViT-B-16-SigLIP'):
        super(SigLIP, self).__init__()
        self.tokenizer = get_tokenizer(model_name)
        self.model = create_model_from_pretrained(model_name)[0]
        self.model.eval()
        del self.model.visual
        self.model = self.model.text
        self.dummy_param = nn.Parameter(torch.empty(0))


    def forward(self, inp, tokenized=True):
        if tokenized:
            encoded_input = inp
        else:
            encoded_input = {'input_ids' : self.tokenizer(inp, context_length=self.model.context_length)}

        # We get the encoded input from the tokenizer as a dictionary with the keys
        # 'input_ids', 'token_type_ids' and 'attention_mask'. We need to remove the
        # 'token_type_ids' and 'attention_mask' keys as they are not used by the model.
        encoded_input = encoded_input['input_ids']
        encoded_input = encoded_input.to(self.dummy_param.device)
        encoded_input_shape = encoded_input.size()
        # If there's a sequence the input might come as a 2d tensor of (batch_size, sequence_length, tokens)
        if len(encoded_input_shape) == 3:
            encoded_input = encoded_input.view(-1, encoded_input.size(-1))

        with torch.no_grad():
            if encoded_input.dim() == 2:
                sentence_embeddings = self.model(encoded_input)
            else:
                sentence_embeddings = self.model(encoded_input.unsqueeze(0))
            # Reshape back to the original shape
            # if len(encoded_input_shape) == 3:
            #     sentence_embeddings = sentence_embeddings.view(encoded_input.size(0), encoded_input.size(1), -1)


        return sentence_embeddings

        
class TextMLP(nn.Module):
    """
    MLP for text embeddings
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128,
            num_layers=2, non_linearity=nn.ReLU()):
        super(TextMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.non_linearity = non_linearity
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, inp, detach_weights=False):
        for i in range(self.num_layers):
            inp = self.layers[i](inp)
            if i != self.num_layers - 1:
                inp = self.non_linearity(inp)
        return inp

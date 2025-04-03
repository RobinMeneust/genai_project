from torch import nn
from transformers import BertTokenizer, BertModel


class FrozenTextEncoder(nn.Module):
    def __init__(self):
        super(FrozenTextEncoder, self).__init__()

        
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained("bert-base-uncased")
        
        # Freeze params
        # TODO: Recheck if this is valid for the loaded model (BERT)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text):
        encoded_input = self._tokenizer(text, return_tensors='pt')
        return self._model(**encoded_input)
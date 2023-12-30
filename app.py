import gradio as gr
import torch
import torch.nn as nn
import spacy

label = {
   0:'bad',
   1:'good'
}

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    L = []
    doc = nlp(text)
    #L = [x.text for x in doc if not x.is_stop or not x.is_punct]
    for token in doc:
        if token.is_punct:
            continue
        else:
             L.append(token.text)
    return " ".join(L).lower()

def num_vec(text):
    doc = nlp(text)
    return doc.vector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MovieReviewClassification(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super().__init__()
    self.M = hidden_dim
    self.L = layer_dim

    self.rnn = nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=layer_dim,
        batch_first=True)
    #batch_first to have (batch_dim, seq_dim, feature_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, X):
    # initial hidden state and cell state
    h0 = torch.zeros(self.L,self.M).to(device)
    c0 = torch.zeros(self.L,self.M).to(device)

    out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))

    # h(T) at the final time step
    out = self.fc(out)
    return out
  
input_dim = 96
hidden_dim = 32
layer_dim = 2
output_dim = 1

model = MovieReviewClassification(input_dim,hidden_dim,layer_dim,output_dim).to(device)

model.load_state_dict(torch.load('final2.pth',map_location=torch.device('cpu')))

def prediction(text):
    preprocessed = preprocess(text)
    input = torch.FloatTensor(num_vec(preprocessed)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input)
        # prob = torch.sigmoid(output)
        pred = torch.round(output)
        return label[int(pred.item())]

   
demo = gr.Interface(fn=prediction, inputs="text", outputs="text",title='Movie Review Classification',article='created by amith')
    
if __name__ == "__main__":
    demo.launch(share=True)   
   

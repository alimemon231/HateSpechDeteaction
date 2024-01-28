import torch
from torch import nn

class LanguageDetectionModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LanguageDetectionModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
            
class UrduHateSpeechModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(UrduHateSpeechModel, self).__init__()  # Use double underscore before "init"
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

# Define a deep feedforward NLP model
class EnglishModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(EnglishModel, self).__init__()
        # Define your model architecture here
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.relu = nn.ReLU()
        for _ in range(num_hidden_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        out = self.output_layer(x)
        return out
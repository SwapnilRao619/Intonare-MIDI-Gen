import torch
import torch.nn as nn
import json
import numpy as np
import music21 as m21
from Preprocess import SEQUENCE_LENGTH, MAPPING_PATH, load, SINGLE_FILE_DATASET
import os
import random

MODEL_PATH = "model.pth"
OUTPUTS_DIR = "outputs"
DURATION = 0.25  # quarter length

# Load the mappings to determine the number of unique classes
with open(MAPPING_PATH, "r") as fp:
    mappings = json.load(fp)

class LSTMModel(nn.Module):
    def __init__(self, output_units, num_units):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=output_units, hidden_size=num_units[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_units[0], hidden_size=num_units[1], batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_units[1], output_units)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return x

def generate_sequence(model, seed, num_steps, output_units, device):
    model.eval()
    seed = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(device)
    seed = torch.nn.functional.one_hot(seed, num_classes=output_units).float()

    generated = []
    for _ in range(num_steps):
        with torch.no_grad():
            output = model(seed)
        probabilities = torch.softmax(output, dim=1).cpu().data.numpy()
        next_step = np.random.choice(range(output_units), p=probabilities.flatten())
        generated.append(next_step)

        next_input = torch.tensor([[next_step]], dtype=torch.long).to(device)
        next_input = torch.nn.functional.one_hot(next_input, num_classes=output_units).float()
        seed = torch.cat((seed[:, 1:, :], next_input), dim=1)
    
    return generated

def decode_sequence(sequence, mappings):
    inv_mappings = {v: k for k, v in mappings.items()}
    decoded_sequence = [inv_mappings[i] for i in sequence]
    return decoded_sequence

def save_midi(sequence, file_path, time_step=DURATION, instrument='Piano', dynamics='mf', articulation='Staccato'):
    stream = m21.stream.Stream()

    # Select the instrument based on user choice
    instruments_dict = {
        'Piano': m21.instrument.Piano(),
        'Violin': m21.instrument.Violin(),
        'Flute': m21.instrument.Flute(),
        'Guitar': m21.instrument.Guitar()
    }
    instrument = instruments_dict.get(instrument, m21.instrument.Piano())
    stream.append(instrument)

    step_duration = time_step
    start_symbol = None
    step_counter = 1
    for i, symbol in enumerate(sequence):
        if symbol == "/":
            continue  # Ignore the delimiter
        if symbol != "_" or i == len(sequence) - 1:
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(
                        int(start_symbol), quarterLength=quarter_length_duration)
                
                # Add dynamics based on user choice
                dynamics_dict = {
                    'p': m21.dynamics.Dynamic("p"),
                    'mf': m21.dynamics.Dynamic("mf"),
                    'f': m21.dynamics.Dynamic("f")
                }
                dynamic = dynamics_dict.get(dynamics, m21.dynamics.Dynamic("mf"))
                m21_event.expressions.append(dynamic)

                # Add articulations based on user choice
                articulations_dict = {
                    'Staccato': m21.articulations.Staccato(),
                    'Tenuto': m21.articulations.Tenuto(),
                    'Accent': m21.articulations.Accent()
                }
                articulation = articulations_dict.get(articulation, m21.articulations.Staccato())
                m21_event.articulations.append(articulation)

                stream.append(m21_event)
                step_counter = 1
            start_symbol = symbol
        else:
            step_counter += 1

    midi_file = m21.midi.translate.music21ObjectToMidiFile(stream)
    midi_file.open(file_path, 'wb')
    midi_file.write()
    midi_file.close()
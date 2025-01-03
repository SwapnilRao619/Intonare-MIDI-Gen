import torch
import torch.nn as nn
import torch.optim as optim
from Preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
import json
import time

# Load the mappings to determine the number of unique classes
with open(MAPPING_PATH, "r") as fp:
    mappings = json.load(fp)

OUTPUT_UNITS = len(mappings)  # Number of unique symbols in the mappings
NUM_UNITS = [256, 256]  # Added another LSTM layer
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.pth"
CLIP = 5  # Gradient clipping

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss_fn=LOSS, learning_rate=LEARNING_RATE):
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMModel(output_units, num_units).to(device)  # Move model to GPU
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = loss_fn

    model.train()
    total_start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)  # Move data to GPU
            data_one_hot = torch.nn.functional.one_hot(data, num_classes=output_units).float()
            optimizer.zero_grad()
            output = model(data_one_hot)
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}')
        scheduler.step()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader)}, Time: {epoch_duration // 60} minutes {epoch_duration % 60} seconds')

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total Training Time: {total_duration // 60} minutes {total_duration % 60} seconds')

    torch.save(model.state_dict(), SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
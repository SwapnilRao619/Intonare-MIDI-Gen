from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import json
import os
import random
from Generator import LSTMModel, generate_sequence, decode_sequence, save_midi

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = "model.pth"
OUTPUTS_DIR = "outputs"
MAPPING_PATH = "mapping.json"
DURATION = 0.25  # quarter length

# Load the mappings to determine the number of unique classes
with open(MAPPING_PATH, "r") as fp:
    mappings = json.load(fp)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    instrument = data.get('instrument', 'Piano')
    dynamics = data.get('dynamics', 'mf')
    articulation = data.get('articulation', 'Staccato')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_units = len(mappings)
    num_units = [256, 256]

    model = LSTMModel(output_units, num_units).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    # Load the dataset and select a seed sequence
    with open("file_dataset", "r") as fp:
        songs = fp.read()
    int_songs = [mappings[symbol] for symbol in songs.split() if symbol != "/"]
    seed_sequence = int_songs[:64]

    generated_sequence = generate_sequence(model, seed_sequence, 500, output_units, device)
    decoded_sequence = decode_sequence(generated_sequence, mappings)

    # Print the generated MIDI representation for debugging
    print("Generated MIDI representation:", decoded_sequence)

    output_path = os.path.join(OUTPUTS_DIR, "generated_song.mid")
    save_midi(decoded_sequence, output_path, time_step=DURATION, instrument=instrument, dynamics=dynamics, articulation=articulation)
    
    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
    app.run(host='0.0.0.0', port=5000)
# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
import pandas as pd
from pathlib import Path
# Torch modules
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch
# Custom modules
import batch_loader
import cnn_architecture
import pre_processing
# App (Gradio) modules
import gradio as gr


download_path = Path.cwd()/'UrbanSound8K'

# Read metadata file
metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

# Construct file path by concatenating fold and file name
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# Take relevant columns
df = df[['relative_path', 'classID']]
print(df.head())

# Assuming SoundDS is already defined and it takes df and data_path as arguments
data_path = download_path/'audio'

SoundDS = batch_loader.SoundDS
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)



## FUNCTIONAL USE

# Create the model and put it on the GPU if available
myModel = cnn_architecture.AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')
  
num_epochs=16   # Tune this to train longer
training(myModel, train_dl, num_epochs)

# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set
inference(myModel, val_dl)

## GRADIO QUICK APP CODE

# Save the trained model
torch.save(myModel.state_dict(), 'audio_classifier_model.pth')
AudioUtil = pre_processing.AudioUtil

model = cnn_architecture.AudioClassifier() 
model.load_state_dict(torch.load('audio_classifier_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Prediction function
class_labels = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

def predict(audio_file_path):
    # Load and preprocess the audio file
    aud = AudioUtil.open(audio_file_path)
    reaud = AudioUtil.resample(aud, 44100)  # Example sample rate
    rechan = AudioUtil.rechannel(reaud, 2)  # Example channel count
    dur_aud = AudioUtil.pad_trunc(rechan, 4000)  # Example duration
    shift_aud = AudioUtil.time_shift(dur_aud, 0.4)  # Example shift
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    # Assuming the model expects a 4D tensor of shape [batch_size, 1, height, width]
    # Modify the line below to correctly reshape the tensor
    processed_audio = aug_sgram.unsqueeze(0)  # This adds the batch dimension
    processed_audio = processed_audio.squeeze(1)  # Example: remove one unnecessary dimension

    # Make a prediction
    with torch.no_grad():
        processed_audio = processed_audio.to(device)
        outputs = model(processed_audio)  # If processed_audio is correctly shaped, no need for unsqueeze(0) now
        _, predicted_class_index = torch.max(outputs, 1)
    
    # Convert the predicted class index to a human-readable label
    predicted_label = class_labels[predicted_class_index.item()]
    return predicted_label

# ... Gradio interface code ...

iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=None, type="filepath"),  # Adjust according to your Gradio version
    outputs="text"
)

# Launch the app
iface.launch()
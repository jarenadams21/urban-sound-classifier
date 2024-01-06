# Simple Deep Learning Sound Classification
This is a simple gradio application that trains on a dataset of urban sounds and classifies between 10 different classes. It then takes microphone input and classifies after the model is done training. (THIS MODEL IS NOT FINELY TUNED, and its training steps do not yield confident accuracy results)

With the help of Ketan Doshi: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
Uses the UrbanSound8K audio dataset of 10 different plausible urban city sounds: https://urbansounddataset.weebly.com/urbansound8k.html

## Future Work
* Modify training to consider how much attention the gun_shot and louder noises are grabbing. In its current state, I think it can sometimes correctly identify the more obvious distinctions among classes , but a lot are getting classified as gun shots or car horns.

```
int main() {
## comrade.py is for debugging purpose for gradio app stuff
## pre_processing.py uses torch audio functions to modify audio clips to be used by the batch_loader
## batch_loader.py is for standardizing the properties of the audio ???<-?(X)>
## cnn_architecture.py sets up the matrices to be used for the training computations
### load_data.py runs the training, inference, and runs the app that uses the output model

return -
return +
return 1
return 0
}

```
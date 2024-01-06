# Simple Deep Learning Sound Classification
This is a simple gradio application that trains on a dataset of urban sounds and classifies between 10 different classes. It then takes microphone input and classifies after the model is done training. (THIS MODEL IS NOT FINELY TUNED, and its training steps do not yield confident accuracy results)

With the help of Ketan Doshi: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
Uses the UrbanSound8K audio dataset of 10 different plausible urban city sounds: https://urbansounddataset.weebly.com/urbansound8k.html

## Future Work
* Modify training to consider how much attention the gun_shot and louder noises are grabbing. In its current state, I think it can sometimes correctly identify the more obvious distinctions among classes , but a lot are getting classified as gun shots or car horns.

### Pillars of Improvement
#### Testing Different Kernel Sizes
You can experiment with different kernel sizes to see how they affect your model's ability to capture features. Common choices are 3x3, 5x5, and sometimes larger like 7x7.

Example modification for different kernel sizes:
```
# First Convolution Block with a different kernel size
self.conv1 = nn.Conv2d(2, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
```

#### Varying the Number of Convolutional Layers
Adding more layers can increase the model's capacity but also risks overfitting if your dataset isn't large enough.

Example modification for adding an extra convolutional layer:
```
# Adding a fifth convolutional layer
self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
self.relu5 = nn.ReLU()
self.bn5 = nn.BatchNorm2d(128)
self.drop5 = nn.Dropout(0.25)
init.kaiming_normal_(self.conv5.weight, a=0.1)
self.conv5.bias.data.zero_()
conv_layers += [self.conv5, self.relu5, self.bn5, self.drop5]

# Don't forget to adjust the linear layer accordingly
self.lin = nn.Linear(in_features=128, out_features=10)  # Adjust the in_features based on your architecture
```

#### Changing Filter Sizes
Increasing the number of filters can help the model to learn more complex features. However, more filters mean more parameters and potentially longer training times and increased risk of overfitting.
```
# Second Convolution Block with more filters
self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# Update the batch norm to match the number of filters
self.bn2 = nn.BatchNorm2d(32)

```

#### Optimization Strategies
* Baseline Model: Start with your current model as a baseline.
* Iterative Changes: Make small, incremental changes. For example, first experiment with different kernel sizes, then try adjusting the number of filters.
* Validation and Comparison: After each change, train the model and evaluate its performance on a validation set.
* Logging Results: Keep detailed records of each experiment - what you changed and how it impacted performance.

#### Misc Model Tuning Routes
1. Stride and Padding: Adjusting these can change how rapidly your model reduces the size of its feature maps and can affect what features the model focuses on.
2. Dropout Rate: Experiment with different dropout rates (e.g., 0.3, 0.5) to see how they affect overfitting.
3. Learning Rate: Small adjustments in the learning rate can have a big impact on performance.
4. Optimizers: Try different optimizers like SGD, RMSprop, etc., to see how they affect training.

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
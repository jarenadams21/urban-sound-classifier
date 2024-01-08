The convolutional layers (nn.Conv2d) in the model have increasing numbers of filters (or channels): from 2 to 8 to 16 to 32, and finally to 64.
This is a common practice in CNN design. The idea is to start with a small number of filters to capture low-level features (like edges or simple textures) and gradually increase the number of filters to capture more complex and abstract features in deeper layers.

## Building Deeper Networks
### Striding
1. Definition: Striding refers to the step size the convolution filter moves across the input image (or feature map). In a typical convolution operation, the stride is set to 1, meaning the filter moves one pixel at a time. However, if you set a larger stride (e.g., 2 or more), the filter skips pixels, effectively reducing the spatial dimensions of the output feature map.

2. Example: If you have a 3x3 filter with a stride of 2 applied to a 10x10 image, the resulting output will be smaller than the original   10x10 size because the filter jumps two pixels at a time instead of one.

3. Effect: Increasing the stride reduces the size of the output feature maps, which can be useful for progressively reducing the spatial dimension of the input as it moves through layers of the network.

### Pooling
1. Definition: Pooling is an operation that summarizes the features in a region of a feature map. The most common type of pooling is max pooling, where a filter (often 2x2) moves across the feature map and takes the maximum value from each region it covers. Average pooling, taking the average value, is another type of pooling.

2. Example: A 2x2 max pooling operation applied to a feature map effectively reduces its width and height by half. If applied to an 8x8 feature map, the resulting output will be 4x4.

3. Effect: Pooling reduces the dimensions of feature maps, which not only helps in reducing computation but also makes the representation somewhat invariant to small shifts and distortions. Additionally, pooling helps in extracting dominant features which are robust to variations in the input.

### Reminders
In each epoch, we go through validation/training data and generate output predictions from the model. Because its probabilistic, i assume pytorch batches this into one request and provides a ratio in some way of correct vs incorrect predictions. This can be used to tally up a running correct vs total predictions count, and help with identifying/validating its performance. The inference at the end acts as a last step to notify you on the state of the model before its loaded and used in the app

### Needs more study
Why this might happen (probably normal, curious on reasoning why certain epochs might be stagnant and if thats avoidable):
```
Epoch 8/16, Train Loss: 0.7759, Train Acc: 0.7365, Val Loss: 0.6746, Val Acc: 0.7761
Epoch 9/16, Train Loss: 0.7513, Train Acc: 0.7419, Val Loss: 0.6257, Val Acc: 0.7944
Epoch 10/16, Train Loss: 0.6793, Train Acc: 0.7677, Val Loss: 0.6003, Val Acc: 0.7950
Epoch 11/16, Train Loss: 0.6389, Train Acc: 0.7836, Val Loss: 0.5752, Val Acc: 0.8167
```

#### Summary
```
/*
In many CNN architectures, pooling layers are interspersed between convolutional layers, and convolutions often have a stride greater than 1 to progressively reduce the spatial dimensions as the network goes deeper. This design pattern helps in building deeper networks that can learn more abstract and complex features.
*/
```
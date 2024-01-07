The convolutional layers (nn.Conv2d) in the model have increasing numbers of filters (or channels): from 2 to 8 to 16 to 32, and finally to 64.
This is a common practice in CNN design. The idea is to start with a small number of filters to capture low-level features (like edges or simple textures) and gradually increase the number of filters to capture more complex and abstract features in deeper layers.

## Building Deeper Networks
### Striding
    Definition: Striding refers to the step size the convolution filter moves across the input image (or feature map). In a typical convolution operation, the stride is set to 1, meaning the filter moves one pixel at a time. However, if you set a larger stride (e.g., 2 or more), the filter skips pixels, effectively reducing the spatial dimensions of the output feature map.

    Example: If you have a 3x3 filter with a stride of 2 applied to a 10x10 image, the resulting output will be smaller than the original   10x10 size because the filter jumps two pixels at a time instead of one.

    Effect: Increasing the stride reduces the size of the output feature maps, which can be useful for progressively reducing the spatial dimension of the input as it moves through layers of the network.

### Pooling
    Definition: Pooling is an operation that summarizes the features in a region of a feature map. The most common type of pooling is max pooling, where a filter (often 2x2) moves across the feature map and takes the maximum value from each region it covers. Average pooling, taking the average value, is another type of pooling.

    Example: A 2x2 max pooling operation applied to a feature map effectively reduces its width and height by half. If applied to an 8x8 feature map, the resulting output will be 4x4.

    Effect: Pooling reduces the dimensions of feature maps, which not only helps in reducing computation but also makes the representation somewhat invariant to small shifts and distortions. Additionally, pooling helps in extracting dominant features which are robust to variations in the input.

#### Summary
In many CNN architectures, pooling layers are interspersed between convolutional layers, and convolutions often have a stride greater than 1 to progressively reduce the spatial dimensions as the network goes deeper. This design pattern helps in building deeper networks that can learn more abstract and complex features.
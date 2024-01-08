# Performance Testing
GOAL: 90% accuracy+

## Training and Val Accuracies

### 1/7/2024
* 16 epochs
* 5 CNN layers (2 (original input), 8, 16, 32, 64, 128 (final output into classifier) )
* Dropout values of 0.5, 0.30, 0.35, 0.40, 0.5 for each layer respectively [NEEDS_UPDATE]
* Kernel size: (5,5), (7,7), (7,7), (7,7), (5,5) > Chose these to gather more information about lower/mid frequency distinctions whereas the input data (after pre-processing) and the final convolution block use smaller kernels [NEEDS_UPDATE]
* Should try to synchronize kernel, convlution layer, and dropout parameters more
```
Epoch 1/16, Train Loss: 2.0230, Train Acc: 0.2662, Val Loss: 1.7203, Val Acc: 0.3562
Epoch 2/16, Train Loss: 1.5743, Train Acc: 0.4499, Val Loss: 1.3427, Val Acc: 0.5063
Epoch 3/16, Train Loss: 1.3287, Train Acc: 0.5331, Val Loss: 1.1236, Val Acc: 0.6266
Epoch 4/16, Train Loss: 1.1717, Train Acc: 0.6018, Val Loss: 1.0288, Val Acc: 0.6569
Epoch 5/16, Train Loss: 1.0663, Train Acc: 0.6351, Val Loss: 0.9002, Val Acc: 0.6930
Epoch 6/16, Train Loss: 0.9495, Train Acc: 0.6785, Val Loss: 0.7493, Val Acc: 0.7434
Epoch 7/16, Train Loss: 0.8493, Train Acc: 0.7070, Val Loss: 0.7342, Val Acc: 0.7468
Epoch 8/16, Train Loss: 0.7759, Train Acc: 0.7365, Val Loss: 0.6746, Val Acc: 0.7761
Epoch 9/16, Train Loss: 0.7513, Train Acc: 0.7419, Val Loss: 0.6257, Val Acc: 0.7944
Epoch 10/16, Train Loss: 0.6793, Train Acc: 0.7677, Val Loss: 0.6003, Val Acc: 0.7950
Epoch 11/16, Train Loss: 0.6389, Train Acc: 0.7836, Val Loss: 0.5752, Val Acc: 0.8167
Epoch 12/16, Train Loss: 0.6173, Train Acc: 0.7834, Val Loss: 0.5347, Val Acc: 0.8150
Epoch 13/16, Train Loss: 0.5698, Train Acc: 0.8083, Val Loss: 0.5348, Val Acc: 0.8219
Epoch 14/16, Train Loss: 0.5606, Train Acc: 0.8092, Val Loss: 0.4809, Val Acc: 0.8419
Epoch 15/16, Train Loss: 0.5383, Train Acc: 0.8102, Val Loss: 0.5461, Val Acc: 0.8144
Epoch 16/16, Train Loss: 0.5114, Train Acc: 0.8239, Val Loss: 0.4964, Val Acc: 0.8408
Finished Training
Accuracy: 0.11, Total items: 1746
# Not sure what this low accuracy is about, but the model performs well in gradio
```


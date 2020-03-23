# PyTorch-CNN-FashionMNIST
PyTorch Convolutional Neural Network trained on FashionMNIST dataset

The process is as follows:
    1. Prepare the data (Extract Transform Load)
        1.1 Extract - Get the FashionMNIST dataset
        1.2 Transform - Put the data into tensor form
        1.3 Load - Put the data into an object to make it easily accessible. Using torchvision and DataLoader packages.
    2. Build the model (network)
        2.1 Network class extends nn.Module
        2.2 Define the layers in the constructor using nn package
        2.3 Implement the forward() method
    3. Train the model
        3.1. Get batch from the training set (using the created DataLoader)
        3.2. Pass batch to network (call the network object directly)
        3.3. Calculate the loss (difference between the predicted values and true values) **LOSS FUNCTION** (using functional package)
        3.4. Calculate the gradient of the loss function w.r.t the network's weights **BACKPROP** (backward() function)
        3.5. Update the weights using the gradients to reduce the loss **OPTIMIZATION ALGORITHM** (using optim package)
        3.6. Repeat steps 1-5 until one epoch is completed (EPOCH=complete pass through all samples of the training set)
        3.7. Repeat steps 1-6 for as many epochs required to obtain the desired level of accuracy.
    4. Test the model
        4.1. Get batch from the test set
        4.2. Pass the batch to the network
        4.3. Compute the loss and accuracy
    5. Analyze the results
        5.1 Create a confusion matrix (sklearn metrics)
        5.2 Track the loss and accuracy
        5.3 Save the results and weights to disk

# TODO
- Add a **RunManager** to factorize the training loop and save results to disk. 
- Test the trained models and analyze the results
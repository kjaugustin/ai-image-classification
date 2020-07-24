# Image classification using machine learning
Images have metadata that indicates its orientation. There are 4 different classes of orientation. They are 0, 90, 180 and 270 degrees. Here our task is to build multiple machine learning models from scratch with Python to classify images in to these 4 categories.

The training dataset consists of 40,000 images and the test dataset consists of 1000 images. Each row in these datasets represent an image and it starts with the image name orientation and then numerical feature vectors that represent red, blue and green pixels in the image. The numerical features are generated by rescaling image to 8x8 pixels, resulting in an 8x8x3 = 192 dimensional feature vector. 

1. Implemented a k-nearest neighbor classifier from scratch to predict the label of each test image. 

To train the model enter the following from the command line: 
```
    ./orient.py train train_file.txt model_file.txt nearest
```
To test the model enter the following from the command line:  
```
    ./orient.py test test_file.txt model_file.txt nearest
```

2. Implemented adaboost classifier from scratch to predict the label of each test image. 

To train the model enter the following from the command line:  
```
    ./orient.py train train_file.txt model_file.txt adaboost
```
To test the model enter the following from the command line:  
```
    ./orient.py test test_file.txt model_file.txt adaboost
```
3. Implemented neural network classifier from scratch to predict the label of each test image. 

To train the model enter the following from the command line:   
```
    ./orient.py train train_file.txt model_file.txt nnet
```
To test the model enter the following from the command line:  
```
    ./orient.py test test_file.txt model_file.txt nnet
```

All these classifiers were ran multiple times with different parameters to choose the best model based on accuracy. The best model parameters are stored in model.txt. 

To train with the best model enter the following from the command line: 
```
    ./orient.py train train_file.txt model_file.txt best
```
To test with the best model enter the following from the command line:  
```
    ./orient.py test test_file.txt model_file.txt best
```

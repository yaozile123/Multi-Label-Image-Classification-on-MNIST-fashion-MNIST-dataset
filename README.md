# mnist-with-error-analysis
## Purpose
Computer vision has become one of the most important area in computer science. During this project, we'll collected and pre-processed images from MNIST and fashion-MNIST images.You will see the performance of each model (Logistic Regression,svm, decision_tree, lightGBM) when they met hand-written/clothes images.In the end, this project will show how to do the error analysis over multi-labels. We will analyzed error on handwritten/clothes images and discovered several images are always easier to mis-classify than other numbers.

## Dependencies
- pandas
- numpy
- requests
- json
- sklearn
- matplotlib
- pytorch(MNIST http://yann.lecun.com/exdb/mnist/ and fashion-MNIST https://github.com/zalandoresearch/fashion-mnist)
- pillow
- plotly
## Collect and Pre-Processed Images
The Mnist database is a large database which contained 70000 images of hand-written numbers(from 0 to 9).We can import the dataset from Pytorch directly. Mnist helped us split the train set and test set already(60000:10000). Here is the overview of the Mnist data set.


<img src= "image/mnist_sample.png">
We could do some pre-processing and fit with the models that we choose.

## Model fitting
### Model choosing
As always, the module sklearn provided us various models to use directly. We will use Logistic Regression,svm, decision_tree, lightGBM at here to see their performance.
- Logistic Regression
<img src= "image/minst_log.png">
- SVM
<img src= "image/mnist_svm.png">
- Decision_Tree
<img src= "image/mnist_tree.png">
- LightGBM
<img src= "image/mnist-lgb.png">

### Error analysis 
Based on the report above, we can see that LightGBM and SVM performed pretty well on Mnist, which reached 96% and 94% accuracy respectively. However, the accuracy of Logistic regression and Decision tree do not meet our expectation. We can compare the predict value and the test set to see what could we do to improve the accuracy. Here is the plot for error anaylysis when using the Logistic Regression model.
<img src= "image/error-analysis.png">
The x-axis represent the error,the y-axis repensted the number of errors for each hand-written digit.For instance, we can find that when we predict number 0, number 8 is the most common error that the model would made.
After switching to decision_tree model, the plot for error analysis should be like:
<img src= "image/error-analysis-tree.png">
In this spot, we can see different model has different xxxxx.
## Why not Mnist?
As you can see, Mnist is a well-established data set and has been overused in the data-science area. We can easily get 96% accuracy without fix any issues.


<img src= "image/fashion_sample.png">

## Results

# SVM-classification-detection (Python2.7)
HoG, PCA, PSO, Hard Negative Mining, Sliding Window, NMS


![image](https://github.com/HansRen1024/SVM-classification-localization/blob/master/example.gif)


Best way to do detection is:

HoG(features) -> PCA(less features) + PSO(best C&gamma) -> origin SVM -> HNM(more features) -> better SVM -> SW -> NMS(bbox regression)

Sorry for my laziness.

I think I should clarify the steps for the program.

1. Extract HoG features (script 1)

2. Train an initial model for pso (script 2)

3. Do pca and pso for better parameters C and gamma (script 6)

4. Use no-pca features and the best parameters to train the second model (script 2)

5. In order to increase the accuracy, use the second model to do hnm and get the final model(script 7)

6. Finally, choose an algorithm you like to do location(script 8 or 9 or 10)

**PS:**

1. The reason I use pca is to accelerate the speed of pso. To be honestly, pso is really slow.

2. For step 4, you can also use features processed by pca, but I strongly advise you to hold as possible as more features. Because more features, higher accuracy.

杯子数据集(Dataset)： https://pan.baidu.com/s/18ho4UI50x4YP6lkrjPm7Kw


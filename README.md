# SVM-classification-detection
HoG, PCA, PSO, Hard Negative Mining, Sliding Window, NMS


Best way to do detection is:

HoG(features) -> PCA(less features) + PSO(best C&gamma) -> origin SVM -> HNM(more features) -> better SVM -> SW -> NMS(bbox regression)


中文地址：http://blog.csdn.net/renhanchi/article/category/7007663

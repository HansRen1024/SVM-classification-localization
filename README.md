# SVM-classification-detection
HoG, PCA, PSO, Hard Negative Mining, Sliding Window, NMS


Best way to do detection is:

HoG(feature) > PCA(less feature) > PSO(C&gamma) > origin SVM > HNM(more feature) > better SVM > SW > SVM > NMS(bbox regression)


中文地址：http://blog.csdn.net/renhanchi/article/category/7007663

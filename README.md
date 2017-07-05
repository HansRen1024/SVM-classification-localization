# SVM-classification-detection
HoG, PCA, PSO, Hard Negative Mining, Sliding Window, NMS


Best way to do detection is:
HoG(feature) > origin SVM > HNM(more feature) > better SVM > SW > SVM > NMS(bbox regression)
      |down          |up
PCA(less feature) > PSO(C&gamma)


中文地址：http://blog.csdn.net/renhanchi/article/category/6974362

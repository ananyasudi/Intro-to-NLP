
In this assignment, Frequency based embeddings using Co-Occurence matrix and SVD and Prediction based Embeddings using CBOW and negative sampling are implemented 


* Frequency-based embeddings use different types of vectors, including Count Vector, TF-IDF Vector, and Co-occurrence Matrix with a fixed context window.
* To get the embeddings, run the command `` python3 svd.py ``

* Prediction-based embeddings, such as Word2vec, use models like Continuous Bag of Words (CBOW) and Skip-Gram (SG). However, the computation of Word2vec's training algorithm can be computationally expensive due to calculating gradients over the entire vocabulary. To address this issue, variants like Hierarchical Softmax output, Negative Sampling, and Subsampling of frequent words were proposed.
* To get the embeddings, run the command `` python3 word2vec.py ``

* word2vec.py - CBOW implementation with nehative sampling
* svd.py -  Frequency based implementation using Co-Occurence matrix and SVD
* embeddings.pt - Saved Pretrained model from CBOW.py
* Report.pdf - Anaylsis of the implementation
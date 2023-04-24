import json
import torch
import nltk
import re
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print which device is used
print(device)
class dataset():
    def __init__(self,path):
        """
        generate vxv matrix from the file 
        reduce the vxv matrix to vxk
        """
       
        vocab=[]
        sentences=[]
        with open(path) as f:
            for line in f:
                try:
                  obj=json.loads(line)
                  text=obj["reviewText"]
                  cleaned_text=clean(text.lower())
                  tokens=tokenise(cleaned_text)
                  # tokens=text.split(" ")
                  sentences.append(tokens)
                  vocab+=[[token] for token in tokens]
                except:
                  pass
        self.vocab = build_vocab_from_iterator(vocab, min_freq=3,
                                           specials=['<UNK>'],
                                           special_first=False)
        self.vocab.set_default_index(self.vocab.get_stoi()['<UNK>'])
        self.index2word={i:word for word,i in self.vocab.get_stoi().items()}
        print(self.vocab.get_stoi())
        print(f"Length of Vocabulary: {len(self.vocab.get_stoi())}")
        # print("Sentences.....")
        # print(sentences)
        # print(self.vocab.get_stoi())
        
        sentences=[[self.vocab[word] for word in sentence] for sentence in sentences]
        len_vocab=len(self.vocab.get_stoi())
        mvv=[[0 for _ in range(len_vocab)] for _ in range(len_vocab)]
        # print("Sentences.....")
        # print(sentences)
        for sentence in sentences:
            for i in range(len(sentence)):
                mini=max(0,i-2)
                maxi=min(len(sentence)-1,i+2)
                for j in range(mini,maxi+1):
                    if j==i: continue
                    mvv[sentence[i]][sentence[j]]+=1
                    
        # print("Matrix of v x v....")
        # print(mvv)#matrix of v x v
        print(f"MVV shape: {len(mvv)}")
        truncatedSVD=TruncatedSVD(80)
        self.u = truncatedSVD.fit_transform(mvv)
        # svd.fit(X)
        # self.u,s,v=np.linalg.svd(mvv)
        
        # self.u=self.u[:,:20] #first 3 columns
        # print(f"Here goes the embedding matrix {self.u}")
    def cosinedistance(self,x,y):
      return (1 - np.dot(x,y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))))
       
    def get_nearest_words(self,word,k):#k nearest words
        word_vec=self.u[self.vocab[word]]
        word_vec_norm=np.linalg.norm(word_vec)
        
        # for i in range(len(self.u)):
        #     print(f"index: {i} word:{self.index2word[i]} vector: {self.u[i]} Cosine Similarity:{np.dot(self.u[i],word_vec)/(np.linalg.norm(self.u[i])*word_vec_norm)}\n")
        
        # sorted_indices=np.argsort([np.dot(word_vec,self.u[i])/(np.linalg.norm(self.u[i])*word_vec_norm) for i in range(len(self.u)) if i!=self.vocab[word] and np.linalg.norm(self.u[i])!=0])
        distances=[(self.cosinedistance(self.u[i],word_vec),self.index2word[i]) for i in range(len(self.u)) if i!=self.vocab[word]]
        distances=sorted(distances)
        
        
        return distances[:k]
        

    def find_embedding(self,word):
        return self.u[self.vocab[word]]


       

def clean(s):
    """
    return text after replacing mentions, hashtags, urls with 
    appropriate placeholders.
    """
    s=s.lower()
    temp = re.sub("((https?\:\/\/)|(www\.))\S+", "<URL>", s)
    temp=re.sub("\s#\S+","<HASHTAG>",temp)
    temp=re.sub("\s@\S+","<MENTION>",temp)
    temp=re.sub("[a-zA-Z0-9]*@\S+","<EMAIL>",temp)
    temp=re.sub("Mr\.","Mr",temp)
    temp=re.sub("Mrs\.","Mrs",temp)
    temp=re.sub("\s+"," ",temp)
    # temp=re.sub("\."," ",temp)

    """
    other tokens like can't , aren't ..can also be replaced
    """
    return temp


def tokenise(s):
    """
    return list of tokens 
    """
    # print(s)
    result=re.findall("<\w+>|\w+|[\.,\"\?\:\;']",s)
    # print(result)
    return result



            
ds=dataset('traindata.json')
print(ds.get_nearest_words('wife',10))
print(ds.get_nearest_words('titanic',10))


from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
words=["wife","moon","tried","forces","they"]
wordvecs=[ds.u[ds.vocab[word]] for word in words]
for i in range(len(words)):
  closest_words=ds.get_nearest_words(words[i],5)
  vecs=[vec[1] for vec in closest_words]
  vectors=[ds.u[ds.vocab[w]] for w in vecs]
  
  # closest_words=closest_vecs[:,1]
  # closest_vecs=closest_vecs[:,0]
  tsne = TSNE(n_components=2, verbose=1, random_state=123,perplexity=50)
  z = tsne.fit_transform(pd.DataFrame(vectors))
  df = pd.DataFrame()
  df["y"] = words[i]
  df["comp-1"] = z[:,0]
  df["comp-2"] = z[:,1]

  sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                  palette=sns.color_palette("hls", 10),
                  data=df)
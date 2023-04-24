import json
import math
import torch
import nltk
import re
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from torch import nn, tensor
import torch.nn.functional as F
import random
LS=nn.LogSigmoid()
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
    temp=re.sub("\."," ",temp)

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
                obj=json.loads(line)
                # print(obj)
                text=obj["reviewText"]
                cleaned_text=clean(text.lower())
                tokens=tokenise(cleaned_text)
                # tokens=text.split(" ")
                sentences.append(tokens)
                vocab+=[[token] for token in tokens]
        self.vocab = build_vocab_from_iterator(vocab, min_freq=1,
                                           specials=['<UNK>','<PAD>'],
                                           special_first=False)
        self.vocab.set_default_index(self.vocab.get_stoi()['<UNK>'])
        print(f"Length of Vocabulary: {len(self.vocab.get_stoi())}")
        
        print(self.vocab.get_stoi())
        
        sentences=[[self.vocab[word] for word in sentence] for sentence in sentences]
        len_vocab=len(self.vocab.get_stoi())
        self.sentences=sentences
        # print("Here goes sentences...")
        # print(self.sentences)
        
    
    def generate_negative(self,contextwindow,target):
        negative=[]
        while len(negative)<2:
            x=random.randint(0,len(self.vocab)-1)
            if x not in contextwindow and x!=target and x not in negative:
                negative.append(x)
        return negative

    # def forward(self,contexts,targets,negatives):

    def get_x_and_y(self):
        # return [[context window],targetword, [negative samples]]
        traindata=[]
        allcontexts=[]
        alltargets=[]
        allnegatives=[]
        c=2 #context window size 
        for sentence in self.sentences:
            for i in range(0,len(sentence)):
                context_window=[]
                mini=max(0,i-2)
                maxi=min(len(sentence)-1,i+2)
                for j in range(mini,maxi+1):
                    if j==i: continue
                    context_window.append(sentence[j])
                context_window=context_window+[self.vocab['<PAD>']]*(4-len(context_window))
                negativesamples=self.generate_negative(context_window,sentence[i])
                single_element=[context_window,[sentence[i]],negativesamples]

                allcontexts.append(context_window)
                alltargets.append(sentence[i])
                allnegatives.append(negativesamples)
                
                traindata.append(single_element)
                # print(single_element)
        batchsize=64
        num_batches=math.ceil(len(allcontexts)/batchsize)
        print(f"Length of training data: {len(traindata)} batchsize: {batchsize} Number of batches: {num_batches}\
            Length of Vocabulary : {len(self.vocab)}")
        batched_data=[]
        for i in range(num_batches):
            # temp=traindata[i:max(i+batchsize,len(traindata))]
            c=allcontexts[i*batchsize:min(i*batchsize+batchsize,len(allcontexts))]
            t=alltargets[i*batchsize:min(i*batchsize+batchsize,len(allcontexts))]
            n=allnegatives[i*batchsize:min(i*batchsize+batchsize,len(allcontexts))]
            batched_data.append([c,t,n])
            # print(f"Number of context windows: {len(c)} {len(t)} {len(n)}\n")

        return batched_data
        

        # batched_data=np.array(batched_data)
        # print([batched_data[0][i][4] for i in range(len(batched_data[0]))]) 

class cbowmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(cbowmodel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self,contexts,targets,negatives):
        contexts_embeds=self.embeddings(tensor(contexts))
        # print(contexts_embeds)
        contexts_embeds=contexts_embeds.mean(dim=1)
        targets_embeds=self.embeddings(tensor(targets))
        negatives_embeds=self.embeddings(tensor(negatives))
        negatives_embeds=negatives_embeds.mean(dim=1)
        # print(contexts_embeds.shape)
        # print(targets_embeds.shape)
        # print(negatives_embeds.shape)
        # pos_loss=sum([np.dot(contexts_embeds[i],targets_embeds[i]) for i in range(len(contexts_embeds))])
        # neg_loss=sum([-1*np.dot(contexts_embeds[i],negatives_embeds[i]) for i in range(len(contexts_embeds))])
        pos_loss=(contexts_embeds*targets_embeds).sum(dim=1)
        neg_loss=(contexts_embeds*negatives_embeds).sum(dim=1)

        pos_loss=LS(pos_loss).mean()
        neg_loss=LS(-neg_loss).mean()
        # print(-(pos_loss+neg_loss))
        return -(pos_loss+neg_loss)


ds=dataset('originaldata.json')
batched_data=ds.get_x_and_y()
cbow=cbowmodel(len(ds.vocab),5)
def trainmodel(model,batched_data,num_epochs):
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    for epoch in range(num_epochs):
        total_loss=0
        for batch in batched_data:
            optimizer.zero_grad()
            # print(f"Lengths {len(batch[0])} {len(batch[1])} {len(batch[2])}")

            loss=model.forward(batch[0],batch[1],batch[2])
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        epochloss=total_loss/len(batched_data)
        print(f'Epoch {epoch+1}, Training loss: {epochloss: .4f}')
trainmodel(cbow,batched_data,5)

embedding_matrix=cbow.embeddings.weight.detach().numpy()
torch.save(cbow.embeddings.weight,'embedding.pt')
# f=open('Embedding_matrix.txt','w')
# for vec in embedding_matrix:
#     print(vec,file=f)
embeddings=torch.load('embedding.pt')
# print(embeddings[9])

# print(embeddings[0])
def cosinedistance(x,y):
      x=x.detach().numpy()
      y=y.detach().numpy()
      return (1 - np.dot(x,y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))))
def getindex(ds,i):
  for word,idx in ds.vocab.get_stoi().items():
    if i==idx:
      return word
  return '<UNK>'
def get_nearest_words(cbow,ds,word,k):
  word_vec=embeddings[ds.vocab[word]]
  word_vec_norm=np.linalg.norm(word_vec.detach().numpy())

  distances=[(cosinedistance(embeddings[i],word_vec),getindex(ds,i)) for i in range(len(embeddings)) if i!=ds.vocab[word]]
  distances=sorted(distances)
  
  return distances[:k]
print(get_nearest_words(cbow,ds,"titanic",10))
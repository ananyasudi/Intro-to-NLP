#calculate n-gram frequencies and find 
#prob of given sentence
import sys
from collections import defaultdict
import random
import math
d=0.75
ngram_freq=[{} for i in range(4)]
ngram_sum_freq=[{} for i in range(4)]
ngram_car=[{} for i in range(4)]
cont_count={}
from tokenisation import clean,tokenise
def calculate_ngramfreq(path):
    textcorpus=open(path,'r')
    alltext=clean(textcorpus.read()) #alltext contains text after replacing urls, mentions, hashtags etc
    # sentences=alltext.split('.')
    sentences=filter(None, alltext.split('.'))
    # print(sentences)
    # print("\n")
    f=open("waste2.txt",'w')
    list_sentences=[]
    """
    first pass- calculate freq of unigrams
    then if any of them has freq==1 replace the word 
    in the sentence array with '<UNK>'
    """
    for sentence in sentences:
        sentence=sentence.replace('\n',' ')
        sentence=tokenise(sentence)
        sen=['<s>','<s>','<s>']
        sen.extend(sentence)
        sen.extend(['</s>'])
        list_sentences.append(sen)
        for word in sen:
            if word in ngram_freq[0]:
                ngram_freq[0][word]+=1
            else:
                ngram_freq[0][word]=1
    ngram_freq[0]['<UNK>']=0
    for sentence in list_sentences:
        for i in range(len(sentence)):
            if ngram_freq[0][sentence[i]]<=1:
                ngram_freq[0][sentence[i]]=0
                sentence[i]='<UNK>'
                ngram_freq[0]['<UNK>']+=1
    for sen in list_sentences:
        sen_len=len(sen)
        for i in range(0,len(sen)):
            if i+1<sen_len:
                if sen[i]+' '+sen[i+1] in ngram_freq[1]:
                    ngram_freq[1][sen[i]+' '+sen[i+1]]+=1
                    ngram_sum_freq[0][sen[i]]+=1
                else:
                    ngram_freq[1][sen[i]+' '+sen[i+1]]=1
                    cont_count[sen[i+1]]=cont_count.get(sen[i+1],0)+1
                    ngram_car[0][sen[i]]=ngram_car[0].get(sen[i],0)+1
                    ngram_sum_freq[0][sen[i]]=ngram_sum_freq[0].get(sen[i],0)+1

            if i+2<sen_len:
                if sen[i]+' '+sen[i+1]+' '+sen[i+2] in ngram_freq[2]:
                    ngram_freq[2][sen[i]+' '+sen[i+1]+' '+sen[i+2]]+=1
                    ngram_sum_freq[1][sen[i]+' '+sen[i+1]]+=1
                else:
                    ngram_freq[2][sen[i]+' '+sen[i+1]+' '+sen[i+2]]=1
                    ngram_car[1][sen[i]+' '+sen[i+1]]=ngram_car[1].get(sen[i]+' '+sen[i+1],0)+1
                    ngram_sum_freq[1][sen[i]+' '+sen[i+1]]=ngram_sum_freq[1].get(sen[i]+' '+sen[i+1],0)+1

            if i+3<sen_len:
                if sen[i]+' '+sen[i+1]+' '+sen[i+2]+' '+sen[i+3] in ngram_freq[3]:
                    ngram_freq[3][sen[i]+' '+sen[i+1]+' '+sen[i+2]+' '+sen[i+3]]+=1
                    ngram_sum_freq[2][sen[i]+' '+sen[i+1]+' '+sen[i+2]]+=1
                else:
                    ngram_freq[3][sen[i]+' '+sen[i+1]+' '+sen[i+2]+' '+sen[i+3]]=1
                    ngram_car[2][sen[i]+' '+sen[i+1]+' '+sen[i+2]]=ngram_car[2].get(sen[i]+' '+sen[i+1]+' '+sen[i+2],0)+1
                    ngram_sum_freq[2][sen[i]+' '+sen[i+1]+' '+sen[i+2]]=ngram_sum_freq[2].get(sen[i]+' '+sen[i+1]+' '+sen[i+2],0)+1
        
    # for ngram in ngram_freq:
    #     for sen,freq in ngram.items():
    #         f.write(str(sen)+str(freq)+"\n")
    # f.write("\n")
    # for ngram in ngram_sum_freq:
    #     for sen,freq in ngram.items():
    #         f.write(str(sen)+str(freq)+"\n")
    # f.write("\n")
    
    # for ngram in ngram_car:
    #     for sen,freq in ngram.items():
    #         f.write(str(sen)+str(freq)+"\n")
def sum_of_c(tokens):
    """
    returns Sum(freq[W_i-n+1:i-1.V])
    """
    n=len(tokens)
    sen_part=' '.join(tokens)
    if sen_part in ngram_sum_freq[n-1]:
        # print("here")
        return [ngram_sum_freq[n-1][sen_part],ngram_car[n-1][sen_part]]
    sen_part=sen_part+' '
    sen_len=len(sen_part)
    freq=0
    num_of_ngrams=0
    for key,count in ngram_freq[n].items():
        if count>0 and len(key)>=sen_len and key[0:sen_len]==sen_part:
        # if sen_part in key:
            # print(f"{sen_part}: {key}")
            freq+=count
            num_of_ngrams+=1
    # if sen_part[:-1] in ngram_sum_freq[n-1]:
    #     print(str(sen_part)+str(ngram_sum_freq[n-1][sen_part[:-1]])+str(" ")+str(ngram_car[n-1][sen_part[:-1]])+" "+str(freq)+" "+str(num_of_ngrams))

    
    return [freq,num_of_ngrams]


f=open("waste.txt",'w')
def p_kn(tokens):
    """
    returns P_kn(Wi|w_i-n-1:i-1) 
    """
    n=len(tokens)
    f.write(str(n)+str(tokens)+'\n')
    if n==1:
        return cont_count.get(tokens[0],1e-6)/len(ngram_freq[1].keys())
    sen_part=' '.join(tokens)
    num=max(ngram_freq[n-1].get(sen_part,0)-d,0)
    [den,count_unique_ngrams]=sum_of_c(tokens[0:n-1])
    if den==0:
        # print(f"Here {tokens}")
        return 0.75*1e-7*p_kn(tokens[1:n])
    lamda=(d*count_unique_ngrams)/den
    # print(tokens)
    return (num/den)+(lamda)*p_kn(tokens[1:n])

def lambda_wb(tokens):
    n=len(tokens)
    sen_part=' '.join(tokens)
    sen_part=sen_part+' '
    sen_len=len(sen_part)
    freq=0
    num_unique=0
    for key,count in ngram_freq[n].items():
        if count>0 and len(key)>=sen_len and sen_part==key[0:sen_len]:
            freq+=count
            num_unique+=1
    return [freq,num_unique]

def p_wb(tokens):
    n=len(tokens)

    if n==1:
        # print(f"{tokens}")
        return ngram_freq[0].get(tokens[0],0)/sum(ngram_freq[0].values())

    [den,num_unique]=sum_of_c(tokens[:-1])#den=sum of frequencies
    sen=' '.join(tokens)
    if den==0:
        return 1e-6*p_wb(tokens[1:n])

    l=den/(den+num_unique)
    first_term=(l*ngram_freq[n-1].get(sen,0)/den)
    second_term=(1-l)*p_wb(tokens[1:n])
    # print(f"{tokens} : {first_term} {second_term}")
    return first_term+second_term


def language_model(sentence,method):

    list_of_tokens=tokenise(clean(sentence))
    
    for i in range(len(list_of_tokens)):
        if list_of_tokens[i] not in ngram_freq[0] or ngram_freq[0][list_of_tokens[i]]<=1:
            list_of_tokens[i]='<UNK>'

    list_of_tokens=['<s>','<s>','<s>']+list_of_tokens+['</s>']
    likely=0
    if method=='k':

        for i in range(3,len(list_of_tokens)):
            likely+=math.log(p_kn(list_of_tokens[i-3:i+1]))
        
    elif method=='w':

        for i in range(3,len(list_of_tokens)):
            likely+=math.log(p_wb(list_of_tokens[i-3:i+1]))
        
    # print(likely)
    
    return math.exp(likely*(-1/(len(list_of_tokens))))
    # print(1/likely)

inp_sen=input("Input Sentence")
method=sys.argv[1]
corpus_path=sys.argv[2]
calculate_ngramfreq(corpus_path)

print(language_model(inp_sen,method))
# calculate_ngramfreq('/home/ananya/NLP/Assignment1/Pride and Prejudice - Jane Austen.txt')
# corpus1=open('/home/ananya/NLP/Assignment1/Pride and Prejudice - Jane Austen.txt','r')
# corpus2=open('/home/ananya/NLP/Assignment1/Ulysses - James Joyce.txt','r')
# """
# creating train1,train2,test1,test2 
# """
# corpus1text=clean(corpus1.read())#after cleaning textcorpus
# corpus1_sen=corpus1text.split('.')
# random.shuffle(corpus1_sen)
# corpus1_test=corpus1_sen[:1000]
# corpus1_train=corpus1_sen[1000:]


# corpus2text=clean(corpus2.read())#after cleaning textcorpus
# corpus2_sen=corpus2text.split('.')
# random.shuffle(corpus2_sen)
# corpus2_test=corpus2_sen[:1000]
# corpus2_train=corpus2_sen[1000:]

# LM1_pride_kn_train=0 #total1 -->pride per1-->kneser ney
# LM1_pride_kn_test=0

# LM2_pride_wb_train=0 #total1 -->pride per1-->kneser ney
# LM2_pride_wb_test=0

# LM3_uly_kn_train=0
# LM3_uly_kn_test=0

# LM4_uly_wb_train=0
# LM4_uly_wb_test=0

# LM1_pride_kn_train_f=open("_LM1_train-perplexity.txt",'w')
# LM2_pride_wb_train_f=open("_LM2_train-perplexity.txt",'w')
# LM1_pride_kn_test_f=open("_LM1_test-perplexity.txt",'w')
# LM2_pride_wb_test_f=open("_LM2_test-perplexity.txt",'w')
# LM3_uly_kn_train_f=open("_LM3_train-perplexity.txt",'w')
# LM4_uly_wb_train_f=open("_LM4_train-perplexity.txt",'w')
# LM3_uly_kn_test_f=open("_LM3_test-perplexity.txt",'w')
# LM4_uly_wb_test_f=open("_LM4_test-perplexity.txt",'w')
# count=0
# for sentence in corpus1_train:
#     LM1_pride_kn_train_f.write(str(sentence)+'\t')
#     LM2_pride_wb_train_f.write(str(sentence)+'\t')
#     per1=language_model(sentence,'k')
#     per2=language_model(sentence,'w')
#     LM1_pride_kn_train_f.write(str(per1)+'\n')
#     LM2_pride_wb_train_f.write(str(per2)+'\n')
#     LM1_pride_kn_train+=per1
#     LM2_pride_wb_train+=per2
#     # print(count)
#     count+=1
# print(f"Train Perplexity- KN- Pride {LM1_pride_kn_train/len(corpus1_train)}")
# print(f"Train Perplexity- wb- Pride {LM2_pride_wb_train/len(corpus1_train)}")
# count=0
# for sentence in corpus1_test:
#     LM1_pride_kn_test_f.write(str(sentence)+'\t')
#     LM2_pride_wb_test_f.write(str(sentence)+'\t')
#     per1=language_model(sentence,'k')
#     per2=language_model(sentence,'w')
#     LM1_pride_kn_test_f.write(str(per1)+'\n')
#     LM2_pride_wb_test_f.write(str(per2)+'\n')
#     LM1_pride_kn_test+=per1
#     LM2_pride_wb_test+=per2
#     # print(count)
#     count+=1
# print(f"Test Perplexity -kn- pride - {LM1_pride_kn_test/len(corpus1_test)}")
# print(f"test Perplexity -wb-Pride - {LM2_pride_wb_test/len(corpus1_test)}")
# calculate_ngramfreq('/home/ananya/NLP/Assignment1/Ulysses - James Joyce.txt')
# count=0
# for sentence in corpus2_train:
#     LM3_uly_kn_train_f.write(str(sentence)+'\t')
#     LM4_uly_wb_train_f.write(str(sentence)+'\t')
#     per1=language_model(sentence,'k')
#     per2=language_model(sentence,'w')
#     LM3_uly_kn_train_f.write(str(per1)+'\n')
#     LM4_uly_wb_train_f.write(str(per2)+'\n')
#     LM3_uly_kn_train+=per1
#     LM4_uly_wb_train+=per2
#     # print(count)
#     count+=1
# print(f"Train Perplexity- KN- ULy {LM3_uly_kn_train/len(corpus2_train)}")
# print(f"Train Perplexity- wb- Uly {LM4_uly_wb_train/len(corpus2_train)}")
# count=0
# for sentence in corpus2_test:
#     LM3_uly_kn_test_f.write(str(sentence)+'\t')
#     LM4_uly_wb_test_f.write(str(sentence)+'\t')
#     per1=language_model(sentence,'k')
#     per2=language_model(sentence,'w')
#     LM3_uly_kn_test_f.write(str(per1)+'\n')
#     LM4_uly_wb_test_f.write(str(per2)+'\n')
#     LM3_uly_kn_test+=per1
#     LM4_uly_wb_test+=per2
#     # print(count)
#     count+=1
# print(f"Test Perplexity- KN- Uly {LM3_uly_kn_test/len(corpus2_test)}")
# print(f"Test Perplexity- wb- Uly {LM4_uly_wb_test/len(corpus2_test)}")
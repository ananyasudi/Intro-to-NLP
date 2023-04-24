import re
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

# textcorpus=open('/home/ananya/NLP/Assignment1/Pride and Prejudice - Jane Austen.txt')
# writefile=open('./waste.txt','w')

# allofit=textcorpus.read()
# cleanedtext=clean(allofit)
# x=tokenise(cleanedtext)
# for i in x:
#     writefile.write(i+'\n')

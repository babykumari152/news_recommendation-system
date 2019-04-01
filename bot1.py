#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nltk
import pandas as pd


# In[12]:


df=pd.read_csv(r'news_articles.csv')


# In[13]:


df


# In[14]:


df['Content'].head(5)


# In[15]:


import string
from nltk.corpus import stopwords


# In[16]:


sp=string.punctuation


# In[17]:


num='0123456789'
sp=sp+num


# In[18]:


sp=list(sp)
# stopwords.words('english')


# In[19]:


from nltk.stem.wordnet import WordNetLemmatizer

lem=WordNetLemmatizer()


# In[20]:


#nltk.download('stem')


# In[21]:


from nltk.stem import PorterStemmer
ps=PorterStemmer


# In[22]:


#nltk.download('wordnet')


# In[23]:


#nltk.download('averaged_perceptron_tagger')


# In[24]:


sno = nltk.stem.SnowballStemmer('english')


# In[25]:


def rem_s(s):
    re= [ss for ss in s if s not in sp]
    re=''.join(re)
    res=[ss for ss in re.split() if ss.lower() not in stopwords.words('english')]
    rel=[sno.stem(w)for w in res]
    
    lems=[lem.lemmatize(s) for s in rel]
    return lems


# In[26]:


df['cont']=df['Content'].apply(rem_s)
# df['cont'] = rem_s(df['Content'])


# In[27]:


cleaned_doc=[each for each in df['cont']]


# In[28]:


cleaned_doc


# In[ ]:


#def rem_s(s):
#    re= [ss for ss in s if s not in sp]
 #   re=''.join(re)
  #  
   # res=[ss for ss in re.split() if ss.lower() not in stopwords.words('english')]
    
    #lems=[lem.lemmatize(s) for s in res]
    #return lems


# In[ ]:


#df['conts']=df['Content'].apply(rem_s)


# In[ ]:


#df['conts']


# In[ ]:


#cleaned_docs=[each for each in df['conts']]


# In[ ]:


######################


# In[ ]:


#final


# In[29]:


import gensim
from gensim.corpora import Dictionary


# In[30]:


#dicts=corpora.Dictionary(cleaned_doc)

dct = Dictionary(cleaned_doc)


# In[31]:


#print(df['cont'])
print(dct)


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


#count_v=CountVectorizer(analyzer=rem_s).fit(df['cont'])


# In[34]:


doc_term_matrix = [dct.doc2bow(doc) for doc in cleaned_doc]


# In[35]:


Lda = gensim.models.ldamodel.LdaModel


# In[36]:


ldamodel = Lda(doc_term_matrix, num_topics=7, id2word = dct, passes=50,minimum_probability=0.0)


# In[37]:


#print(ldamodel['dct.doc2bow(doc)'])
dos_m=ldamodel[doc_term_matrix]


# In[38]:


ldamodel.print_topics()


# In[39]:


d_m=list(dos_m)


# In[40]:


type(d_m)


# In[41]:


#d_m.create_dummies()


# In[42]:


len(d_m)


# In[44]:


#x=np.array([[l[1]for l in each]for each in d_m])


# In[45]:


#x


# In[46]:


topic=[]
probability=[]
for each in d_m:
    t=[]
    p=[]    
    for e in each:
        t.append(e[0])
        p.append(e[1])
    topic.append(t)
    probability.append(p)
    


# In[47]:


import numpy as np


# In[48]:


topic_a=np.array(probability,dtype='float32')


# In[49]:


topic_a


# In[55]:


#df['prob']=topic_a


# In[51]:


#df['probab']=topic_a
p=[]
for each in range(len(topic_a)):
    p.append(topic_a[each])
    #l=([topic_a[each][0],topic_a[each][1],topic_a[each][2],topic_a[each][3],topic_a[each][4],topic_a[each][5],topic_a[each][6]])
    #df['pr']=l
    


# In[54]:


#for each in p:
 #   df['prob']=each


# In[53]:


df['prob']=p


# In[ ]:


df.head()


# In[ ]:


#np.argmax(topic_a[0][2])
topic_a.shape


# In[56]:


df.to_pickle('user_p.csv')


# In[57]:


p=pd.read_pickle('user_p.csv')


# In[58]:


ps=np.argmax(topic_a,axis=1)


# In[59]:


p


# In[60]:


ps.shape


# In[61]:


#def article_to_topic():
df['topic']=ps
    


# In[62]:


df.to_csv('myfiles.csv')


# In[ ]:


#plt.scatter(topic_a[0:4],topic_a[5:9],c='black',s=9)


# In[ ]:


#print(topic_a)


# In[ ]:


#df['Title','Content'].groupby([df['topic']])


# In[ ]:


#topic_a.shape


# In[ ]:


#import matplotlib.pyplot as plt


# In[ ]:


#dd=df['Content'].groupby([df['topic']])


# In[ ]:


#df.DataFrame()
#df


# In[ ]:


#df.groupby(['topic']).apply(mm)


# In[ ]:


#def mm(s):
    


# In[63]:


#count_v.get_feature_names()[4450]
def get_lda_topics(model,num_topics):
    word_dict={}
    for i in range(num_topics):
        words=model.show_topic(i,topn=20)
        word_dict['topic'+str(i)]=[i[0] for i in words]
    return pd.DataFrame(word_dict)    


# In[64]:


gf=get_lda_topics(ldamodel,7)


# In[65]:


#text=[' '.join(each) for each in df['cont'].values]
#text


# In[66]:


gf


# In[ ]:


#from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


#c_transformer = CountVectorizer()
#X_train_tf = c_transformer.fit_transform(text)


# In[ ]:


#X_train_tf.toarray()


# In[ ]:


#print(X_train_tf.toarray()[79])


# In[ ]:


#c=np.array([[2.3866116e-01, 5.4654566e-04, 5.4651889e-04, 4.9632620e-02, 5.4658647e-04
 #4926827e-03, 2.1068338e-02 ,1.4926214e-03]])


# In[67]:


from sklearn.cluster import KMeans


# In[68]:


kmean=KMeans(n_clusters=5)
f=kmean.fit(topic_a)


# In[69]:


a=f.labels_


# In[70]:


mm=df.to_csv('myfile.csv')


# In[71]:


df['cluster']=a


# In[72]:


p=np.random.randint(0,5)
p


# In[73]:


m=range(0,10)


# In[75]:



def bot1(df,n):
    i=np.random.randint(1,100,1)
    #n=np.random.randint(0,5,1)
    #print(df.ix[df['cluster']==n,'Content'].iloc[1])
    num=0
    art=[]
    while(num<n):
        #fo j in range(0,5):
        if (num<=4):
            #print('=============')
            dds=df.ix[df['cluster']==num,'Content'].iloc[i]
            #art.append(dds)
            print(dds)
            #num=num+1
        else:
            dds=df.ix[df['cluster']==np.random.randint(0,4.1),'Content'].iloc[i]
            print(dds)
        num=num+1
      


# In[76]:


bot1(df,7)


# In[77]:


l=[877,16,45,632,677,877]


# In[78]:


df1=pd.DataFrame()


# In[79]:


df1['Article_Id']=l


# In[83]:


df1['session']=1


# In[82]:


df1['user_id']=1


# In[84]:


df1.to_pickle('sesson1.csv')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd


# In[208]:


import math


# In[2]:


import numpy as np


# In[3]:


data=pd.read_pickle('user_p.csv')


# In[4]:


data.head(2)


# In[5]:


def word_count(d):
    return len(d)/5


# In[6]:


#['a_time']
data['weight']=data['user_time']/data['a_time']
data['weights']=data['weight'].apply(lambda x:1/(1+math.exp(-x)))


# In[7]:



data['a_time']=data['Content'].apply(word_count)


# In[8]:


#data['a_time']


# In[9]:


user_article=[8,29,489]
user_time=[10,250,230]


# In[10]:


time_ratios=[]
topic_p=[]


# In[11]:


data[data["Article_Id"]==1]["a_time"].values


# In[12]:


for each in range(len(user_article)):
    t=data.loc[data['Article_Id']==user_article[each],'a_time']
    time_ratio=user_time[each]/t
    time_ratios.append(time_ratio.values)
    
    
    


# In[13]:


time_ratios


# In[14]:


topic=[]
for each in range(len(user_article)):
    t=[]
    #for eac in range(0,7):
    #print('t#'+str(each))
    t.append(data.loc[data['Article_Id']==user_article[each]]['prob'].values)
    topic.append(t)    
    


# In[15]:


topics=np.array(topic)


# In[16]:


topics.shape


# In[17]:


time_r=np.array(time_ratios)


# In[18]:


import math


# In[19]:


time=[]


# In[20]:


time=[1/(1+math.exp(-time_r[each])) for each in range(3)]


# In[21]:


time


# In[22]:


time_e=np.array(time)


# In[23]:


time_r=time_e.reshape(1,3)


# In[24]:


print(time_r.shape)


# In[25]:


topics=topics.reshape(3,1)


# In[26]:


topics


# In[27]:


f=[]
for i in range(len(topics)):
    ts=[]
    #p=topics[i].tolist()
    #print(p)
    for each in topics[i]:
        #print(each)
        ts.append(each*time[i])
    f.append(ts)    
        


# In[28]:


f


# In[29]:


final=np.array(f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


finals=np.sum(final,axis=0)


# In[31]:


user_profile=finals


# In[32]:


user_profile


# In[33]:


user_profile=user_profile.reshape(1,7)


# In[34]:


from sklearn.metrics.pairwise import cosine_similarity


# In[35]:


b=np.array([data['prob']])


# In[36]:


b.shape


# In[37]:


#  
    


# In[38]:


#topicss


# In[39]:


#tos=np.array(topicss)


# In[40]:


#tos


# In[41]:


#tt=np.array(topicss)


# In[42]:


tt=b.reshape(4831,7)


# In[43]:


#topicsss=topicsss.reshape(4831,7)


# In[44]:


tt


# In[45]:


datas=cosine_similarity(tt,user_profile.reshape(1,-1)).flatten()


# In[46]:


print(datas)


# In[47]:


#


# In[48]:


type(datas)


# In[49]:


#finals=np.array(f)
bot2=np.argsort(datas)[-10::]


# In[50]:


#print(topic[0][1])
bot2


# In[51]:


for each in range(len(bot2)):
    bb=data.loc[data['Article_Id']==bot2[each],'Content']
    print(bb)
    
    #print(np.array([final[each]]))
#final[0][0]   


# In[52]:


#user_profile=[]
#for each in range(0,7):
    #app=[final[0][each] ,final[1][each] , final[2][each]]
    #user_profile.append(app)
    #user_profile.append(app)


# In[53]:


#user_p=[]
#for each in user_profile:
    #sums=0
    #for e in each:
    #    sums=sums+e
    #user_p.append(sums)    


# In[54]:


#user_p


# In[55]:


s=np.random.binomial(1,.3,10)


# In[56]:


s


# In[57]:


t1=np.random.normal(loc=200,scale=50,size=1831)


# In[58]:


t2=np.random.normal(loc=300,scale=50,size=1500)


# In[59]:


t3=np.random.normal(loc=70,scale=30,size=1500)


# In[60]:


t1


# In[61]:


t4=np.append(t1,t2)


# In[62]:


t5=np.append(t4,t3)


# In[63]:


t5.shape


# In[64]:


data['user_time']=t5


# In[65]:


session_df=pd.DataFrame()


# In[66]:


session_df['Article_Id']=bot2


# In[67]:


session_df['Article-clicked']=s


# In[68]:


session_df['session']=2


# In[69]:


session_df['user_id']=1


# In[70]:


session_df['time_spent']=0


# In[71]:


session=session_df


# In[72]:


session


# In[73]:


def get_time(df1,df2):
    click_list=df1.loc[df1['Article-clicked']==1,'Article_Id']
    time_s=[]
    for each in click_list:
        time_s.append(df2.loc[df2['Article_Id']==each,'user_time'].values)
    return time_s    
        


# In[74]:


time_sp=get_time(session_df,data)


# In[75]:


session_df


# In[196]:


df1=session_df[session_df['Article-clicked']==1]
df2=session_df[session_df['Article-clicked']==0]


# In[197]:


#df1.drop('time_spent',axis=1)
#df2.drop('time_spent',axis=1)
l1=[]
l2=[]
l3=[]
k=[]
for each in df1['Article_Id']:
    l1.append(data.loc[data['Article_Id']==each,'a_time'].values)
    l2.append(data.loc[data['Article_Id']==each,'user_time'].values)
    k.append(data.loc[data['Article_Id']==each,'weights'].values)
for each in df2['Article_Id']:
    l3.append(data.loc[data['Article_Id']==each,'a_time'].values)
    


# In[198]:


df1['expected_time']=l1
df1['observed_time']=l2
df1['weights']=k
df2['observed_time']=0
df2['expected_time']=l3
df2['weights']=0


# In[199]:


#time=[]
#time.append(data[data['Article_Id']==session_df['Article_Id']]['a_time'])
for each in session['Article_Id']:
    print(each)


# In[80]:


#def get_times(df1,df2):
#    time_e=[]
 #   time_s=[]
  #  for each in session['Article_Id']:
        #print(df1[df1['Articles-clicked']==
#         if df1['Article_Id']==each and  df1['Article-clicked']==1:
#             time_s.append(df2.loc[df2['Article_Id']==each,'user_time'])
#         else:
#             time_s.append(0)
#         time_e.append(df2.loc[df2['Article_Id'],'a_time'])  
#     return time_e,time_s    
    


# In[81]:


#expected_t,spend_t=get_times(session,data)


# In[200]:


df3=pd.concat([df1,df2],ignore_index=True)


# In[201]:


#df3=df3.drop('time_spent',axis=1)
df3


# In[185]:


df=pd.read_pickle('sesson1.csv')


# In[186]:


#df3=df3.drop('wights',axis=1)


# In[86]:


lis=np.random.binomial(1,.3,6)


# In[87]:


lis


# In[187]:


df['Article-clicked']=lis


# In[188]:


df


# In[189]:


df1=df[df['Article-clicked']==1]
df2=df[df['Article-clicked']==0]
l=[]
m=[]
n=[]
k=[]
for each in df1['Article_Id']:
    l.append(data.loc[data['Article_Id']==each,'a_time'].values)
    m.append(data.loc[data['Article_Id']==each,'user_time'].values)
    k.append(data.loc[data['Article_Id']==each,'weights'].values)

for each in df2['Article_Id']: 
    n.append(data.loc[data['Article_Id']==each,'a_time'].values)
df1['expected_time']=l
df1['observed_time']=m
df1['weights']=k
df2['expected_time']=n
df2['observed_time']=0
df2['weights']=0


# In[190]:


session1=pd.concat([df1,df2],ignore_index=True)


# In[194]:


session1


# In[202]:


session=pd.concat([session1,df3],ignore_index=True)


# In[204]:



        
session=session.drop('time_spent',axis=1)


# In[103]:


#def weight_cal(df1,df2):
 #   p= df1/df2
  #  return p
    
    


# In[205]:


session


# In[105]:


data['weight']=data['user_time']/data['a_time']


# In[107]:


#import math


# In[125]:


#session['weight']=session['observed_time']/session['expected_time']


# In[139]:


#session['weights']=session['weight'].apply(lambda x:1/(1+math.exp(-x)))


# In[127]:


#data['weights']=data['weight'].apply(lambda x:1/(1+math.exp(-x)))


# In[206]:


session
#data['weight']


# In[143]:


session.loc[session['Article-clicked']==0]['weightss']=0


# In[182]:


session


# In[209]:


data['prob']


# In[281]:


def make_user_p(df,data):
    df2=df[df['Article-clicked']==1]
    l=[]
    m=[]
#n=[]
#k=[]
    for each in df2['Article_Id']:
        l.append(data.loc[data['Article_Id']==each,'prob'].values)
        m.append(df.loc[df['Article_Id']==each,'weights'].values)
    #k.append(data.loc[data['Article_Id']==each,'weights'].values)
    return l,m

    #for each in df2['Article_Id']: 
    #n.append(data.loc[data['Article_Id']==each,'a_time'].values)
#df1['expected_time']=l
#df1['observed_time']=m
#df1['weights']=k
#df2['expected_time']=n
#df2['observed_time']=0
#df2['weights']=0session):
    #df1=df[df['Article-clicked']==1]
#


# In[309]:


user_int,weight=make_user_p(session,data)


# In[318]:


pp=[]
for each in range(len(user_int)):
    
    for i in user_int[each]:
        s1=[i[j]*weight[each] for j in range(7)]
        print(s1)
    pp.append(s1)
    


# In[320]:


user_profile=np.array(pp)


# In[323]:


user_profile=user_profile.reshape(4,7)


# In[329]:


s=np.sum(user_profile,axis=0)


# In[330]:


s


# In[301]:


weight=np.array((weight))
user_int=np.array((user_int))
for each in user_int[0]:
    print(each)


# In[302]:


user_int.shape


# In[292]:


#da=[]
#for each in user_int:
 #   d=[]
  #  for i in range(7):
   # d.append([each]*weight[each])
    #da.append(d)    


# In[273]:


#y=np.sum(a1,a2)


# In[278]:


a1.shape


# In[263]:


len(y)


# In[332]:


#for ec in range(len(user_int)):
#ou=cosine_similarity(tt,y.reshape(1,-1)).flatten() 
dats=cosine_similarity(tt,s.reshape(1,-1)).flatten()


# In[334]:


bot3=np.argsort(dats)[-10::]


# In[335]:


bot3


# In[336]:


for each in bot3:
    s=data.loc[data['Article_Id']==each,'Content']
    print(s)


# In[ ]:





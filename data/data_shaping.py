#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[22]:


df_market = pd.read_csv("market_data_revise.csv",sep=';',index_col=0)


# In[23]:


df_market


# In[24]:


df_market["USGB10"] = df_market["USGB10"].str.replace('%', '').astype(float)
df_market["Inflation"] = df_market["Inflation"].str.replace('%', '').astype(float)


# In[25]:


type(df_market["USGB10"][0])


# In[26]:


type(df_market["US_SP500"][0])


# In[27]:


type(df_market["Inflation"][100])


# In[30]:


df_market


# In[28]:


df_market.to_csv("market_data.csv")


# In[37]:


df_market.drop(columns="Inflation").to_csv("market_data_rm_inflation.csv")


# In[31]:


df_economy = pd.read_csv("economy_data_revise.csv",sep=';',index_col=0)


# In[32]:


df_economy


# In[34]:


df_economy["pce_core_deflator"] = df_economy["pce_core_deflator"].str.replace('%', '').astype(float)
df_economy["unemployment_rate"] = df_economy["unemployment_rate"].str.replace('%', '').astype(float)


# In[35]:


df_economy


# In[36]:


df_economy.to_csv("economy_data.csv")


# # ここから日本国債

# In[220]:


df_jgbcm = pd.read_csv("jgbcm_all_utf.csv")


# In[221]:


df_jgbcm


# In[222]:


for i in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,40]:
    print(i, type(df_jgbcm[str(i)][10]))


# In[223]:


df_jgbcm.replace("-","")


# In[224]:


for i in [1,2,3,10,15,20,25,30,40]:
    df_jgbcm = df_jgbcm.drop(columns=str(i))


# In[225]:


df_jgbcm


# In[226]:


df_jgbcm["date"] = df_jgbcm["date"].replace("\.","-", regex=True)


# In[227]:


for s in df_jgbcm["date"]:
    print(s)


# In[228]:


print(df_jgbcm["date"][8000])
print(df_jgbcm["date"][4000])


# In[229]:


for i in range(49,65):
    df_jgbcm["date"] = df_jgbcm["date"].replace("S"+str(i), str(1925 + i), regex=True)


# In[230]:


for s in df_jgbcm["date"]:
    print(s)


# In[231]:


print(df_jgbcm["date"][8000])
print(df_jgbcm["date"][4000])


# In[232]:


for i in range(11, 32):
    year = 1988 + i
    df_jgbcm["date"] = df_jgbcm["date"].replace("H"+str(i), str(year), regex=True)


# In[233]:


for i in range(1, 10):
    year = 1988 + i
    df_jgbcm["date"] = df_jgbcm["date"].replace("H"+str(i), str(year), regex=True)


# In[234]:


print(df_jgbcm["date"][8000])
print(df_jgbcm["date"][4000])


# In[235]:


for s in df_jgbcm["date"]:
    print(s)


# In[236]:


for i in range(1,3):
    df_jgbcm["date"] = df_jgbcm["date"].replace("R"+str(i),str(2018 + i), regex=True)


# In[237]:


for s in df_jgbcm["date"]:
    print(s)


# In[238]:


for i in range(1,10):
    df_jgbcm["date"] = df_jgbcm["date"].replace("-"+str(i)+"-", "-0"+str(i)+"-", regex=True)


# In[239]:


for s in df_jgbcm["date"]:
    print(s)


# In[240]:


for i in range(1,10):
    df_jgbcm["date"] = df_jgbcm["date"].replace("-"+str(i)+"$", "-0"+str(i), regex=True)


# In[241]:


for s in df_jgbcm["date"]:
    print(s)


# In[242]:


df_jgbcm


# In[243]:


df_jgbcm = df_jgbcm.set_index("date")


# In[244]:


df_jgbcm


# In[245]:


df_jgbcm.to_csv("jgbcm_4to9.csv")


# In[246]:


for i in [4,5,6,7,8]:
    df_jgbcm = df_jgbcm.drop(columns=str(i))


# In[247]:


df_jgbcm.to_csv("jgbcm_9.csv")


# In[249]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'data_shaping.ipynb'])


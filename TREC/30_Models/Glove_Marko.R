#INITIALIZE

install.packages("word2vec")
install.packages("text2vec")
install.packages("tidyverse")
install.packages("tidytext")
install.packages("stringr")
install.packages("tibble")
install.packages("NLP")
install.packages("dplyr")
install.packages("broom")
install.packages("tokenizers")

library(word2vec)
library(text2vec)
library(tidyverse)
library(tidytext)
library(stringr)
library(tibble)
library(NLP)
library(dplyr)
library(broom)
library(tokenizers)

#LOAD DATA

setwd("/Users/free/Desktop/TREC_Distributed_Machine_Learning-master/TREC/10_Data/30_Balanced Tweets (Crit = High = Medium = Low)/ALL")

train2018eqk <- read.csv("earthquake_TREC_2018_train_BALANCED.csv",header=TRUE)
test2018eqk <- read.csv("earthquake_TREC_2018_test_BALANCED.csv",header=TRUE)
train2018fld <- read.csv("flood_TREC_2018_train_BALANCED.csv",header=TRUE)
test2018fld <- read.csv("flood_TREC_2018_test_BALANCED.csv",header=TRUE)
#combine dataframes
eqk2018<-rbind(train2018eqk,test2018eqk)
fld2018<-rbind(train2018fld,test2018fld)

#CLEAN TEXT

eqk2018_01<-tibble(ID=eqk2018$ID,text=as.character(eqk2018$Tweet),target=eqk2018$Priority)
eqk2018_02<-as.vector(eqk2018_01)
eqk2018_03<-eqk2018_02%>%
  unnest_tokens(word,text)
eqk2018_04<-eqk2018_03%>%
  anti_join(stop_words)

fld2018_01<-tibble(ID=fld2018$ID,text=as.character(fld2018$Tweet),target=fld2018$Priority)
fld2018_02<-as.vector(fld2018_01)
fld2018_03<-fld2018_02%>%
  unnest_tokens(word,text)
fld2018_04<-fld2018_03%>%
  anti_join(stop_words)

#MAKE TERM CO-OCCURENCE MATRIX

list_eqk2018<-list(eqk2018_04$word)
it_eqk2018<-itoken(list_eqk2018,progressbar=FALSE)
vocab_eqk2018<-create_vocabulary(it_eqk2018)
vectorizer_eqk2018<-vocab_vectorizer(vocab_eqk2018)
tcm_eqk2018<-create_tcm(it_eqk2018,vectorizer_eqk2018,skipgrams_window=1L)

list_fld2018<-list(fld2018_04$word)
it_fld2018<-itoken(list_fld2018,progressbar=FALSE)
vocab_fld2018<-create_vocabulary(it_fld2018)
vectorizer_fld2018<-vocab_vectorizer(vocab_fld2018)
tcm_fld2018<-create_tcm(it_fld2018,vectorizer_fld2018,skipgrams_window=1L)

#MAKE GLOVE MODEL FUNCTION

glove_eqk2018=GlobalVectors$new(rank=100,x_max=10)
glove_eqk2018$fit_transform(tcm_eqk2018,n_iter=20)

glove_fld2018=GlobalVectors$new(rank=100,x_max=10)
glove_fld2018$fit_transform(tcm_fld2018,n_iter=20)

#MAKE GLOVE WORD VECTORS

glove_eqk2018_vectors_main<-glove_eqk2018$fit_transform(tcm_eqk2018,n_iter=20,convergence_tol=-1)
glove_eqk2018_vectors_context<-glove_eqk2018$components
glove_eqk2018_vectors_0<-glove_eqk2018_vectors_main+t(glove_eqk2018_vectors_context)

glove_fld2018_vectors_main<-glove_fld2018$fit_transform(tcm_fld2018,n_iter=20,convergence_tol=-1)
glove_fld2018_vectors_context<-glove_fld2018$components
glove_fld2018_vectors_0<-glove_fld2018_vectors_main+t(glove_fld2018_vectors_context)

#EXPORT CSV

df_glove_eqk2018<-as.data.frame(glove_eqk2018_vectors_0)
df_glove_eqk2018_01<-rownames_to_column(df_glove_eqk2018)
de<-df_glove_eqk2018_01
cv2018eqk<-paste(de$V1,de$V2,de$V3,de$V4,de$V5,de$V6,de$V7,de$V8,de$V9,de$V10,de$V11,de$V12,de$V13,de$V14,de$V15,de$V16,de$V17,de$V18,de$V19,de$V20,de$V21,de$V22,de$V23,de$V24,de$V25,de$V26,de$V27,de$V28,de$V29,de$V30,de$V31,de$V32,de$V33,de$V34,de$V35,de$V36,de$V37,de$V38,de$V39,de$V40,de$V41,de$V42,de$V43,de$V44,de$V45,de$V46,de$V47,de$V48,de$V49,de$V50,sep=" ")
write_csv(df_glove_eqk2018_01,"/Users/free/Desktop//GLoVE_eqk2018_100.csv")

df_glove_fld2018<-as.data.frame(glove_fld2018_vectors_0)
df_glove_fld2018_01<-rownames_to_column(df_glove_fld2018)
de<-df_glove_fld2018_01
cv2018fld<-paste(de$V1,de$V2,de$V3,de$V4,de$V5,de$V6,de$V7,de$V8,de$V9,de$V10,de$V11,de$V12,de$V13,de$V14,de$V15,de$V16,de$V17,de$V18,de$V19,de$V20,de$V21,de$V22,de$V23,de$V24,de$V25,de$V26,de$V27,de$V28,de$V29,de$V30,de$V31,de$V32,de$V33,de$V34,de$V35,de$V36,de$V37,de$V38,de$V39,de$V40,de$V41,de$V42,de$V43,de$V44,de$V45,de$V46,de$V47,de$V48,de$V49,de$V50,sep=" ")
write_csv(df_glove_fld2018_01,"/Users/free/Desktop//GLoVE_fld2018_100.csv")

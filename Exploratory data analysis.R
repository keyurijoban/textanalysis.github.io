#Call for the required library
library(NLP)
library(RWeka)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(RTextTools)
library(caret)
library(ggrepel)
library(stringr)

#Set working directory
getwd()
setwd("C:\\xxxx\\xxxx")

#read data
flipkart<-read.csv("flipkart.csv")
dim(flipkart)
str(flipkart)
names(flipkart)
class(flipkart)

#keeping on text variable
flipkart<-data.frame(flipkart)
names(flipkart)
flipkart$text <- as.character(flipkart$text)

#creating hashtag frequency table
hashtag <- str_extract_all(flipkart$text, "#\\S+")
tag_corpus<-Corpus(VectorSource(hashtag))
tag_tdm <- TermDocumentMatrix(tag_corpus)
inspect(tag_tdm[1:10,1:10])
dim(tag_tdm)
hash.freq<-rowSums(as.matrix(tag_tdm))
hash.freq<-subset(hash.freq,hash.freq>=1)
df_hash<-data.frame(term=names(hash.freq),freq=hash.freq)
wordFreq_hash<-df_hash[order(df_hash$freq,decreasing = T),]
write.csv(wordFreq_hash,"hashtag.csv")

#convert to lower case
my_corpus<-Corpus(VectorSource(flipkart$text))
my_corpus<-tm_map(my_corpus,content_transformer(tolower))
my_corpus<-tm_map(my_corpus,removeNumbers)
removeURL <- function(x) gsub("(http[^ ]*)", "", x)
my_corpus<-tm_map(my_corpus,content_transformer(removeURL))
#remove anything other than English letters or space
removeNumPunct<-function(x) gsub("[^[:alpha:][:space:]]*","",x)
my_corpus<-tm_map(my_corpus,content_transformer(removeNumPunct))
#remove stopwords
myStopwords<-c(setdiff(stopwords('english'),c("r","big")),"use","see","used","via","will","amp")
my_corpus<-tm_map(my_corpus,removeWords,myStopwords)
#remove extra whitespace
my_corpus<-tm_map(my_corpus,stripWhitespace)
#Stem words in corpus
my_corpus<-tm_map(my_corpus,stemDocument,language = "english")

#build Term Document Matrix
tdm<-TermDocumentMatrix(my_corpus)
dim
#Use 99% sparsity
tdm <- removeSparseTerms(tdm,.99)
dim(tdm)
inspect(tdm[1:10,1:10])
names(tdm)

#find correlation between terms
cor_2 <- cor(as.matrix(t(tdm)))
dim(cor_2)
View(cor_2)
write.csv(cor_2,"corr.csv")

#inspect frequent words
(freq.terms<-findFreqTerms(tdm,lowfreq = 47))
length(freq.terms)
term.freq<-rowSums(as.matrix(tdm))
term.freq<-subset(term.freq,term.freq>=1)
df<-data.frame(term=names(term.freq),freq=term.freq)
wordFreq<-df[order(df$freq,decreasing = T),]
write.csv(wordFreq,"wordfreq.csv")
dim(wordFreq)

#finding association between terms/words
findAssocs(tdm,'flipkart',0.1)
findAssocs(tdm,'deal',0.2)
findAssocs(tdm,'amazon',0.1)
findAssocs(tdm,'hyderabad',0.1)
findAssocs(tdm,'flipkartbigsal',0.15)
findAssocs(tdm,'bigsal',0.1)
findAssocs(tdm,'deal',0.3)

#wordcloud
#plot word cloud
wordcloud(my_corpus,min.freq =47,max.words = 5067,random.order = F,rot.per=0.3,colors=brewer.pal(8, "Dark2"))

#bigrammatrix
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm_2 <- TermDocumentMatrix(my_corpus, control = list(tokenize = BigramTokenizer))
inspect(tdm_2[1:10,1:10])
names(tdm_2)

#Bigram frequency
(freq.terms2<-findFreqTerms(tdm_2,lowfreq = 100))
length(freq.terms2)
term.freq2<-rowSums(as.matrix(tdm_2))
term.freq2<-subset(term.freq2,term.freq2>=100)
df2<-data.frame(term=names(term.freq2),freq=term.freq2)
wordFreq2<-df2[order(df2$freq,decreasing = T),]
write.csv(wordFreq2,"top_bigram.csv")

#Bigram wordcloud
wordcloud(df2$term,df2$freq,min.freq = 100,random.order = F,rot.per=0.3,colors=brewer.pal(8, "Dark2"))

#trigram matrix
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm_3 <- TermDocumentMatrix(my_corpus, control = list(tokenize = TrigramTokenizer))
inspect(tdm_3[1:10,1:10])

#Trigram frequency
(freq.terms3<-findFreqTerms(tdm_3,lowfreq = 100))
length(freq.terms3)
term.freq3<-rowSums(as.matrix(tdm_3))
term.freq3<-subset(term.freq3,term.freq3>=100)
df3<-data.frame(term=names(term.freq3),freq=term.freq3)
wordFreq3<-df3[order(df3$freq,decreasing = T),]
wordFreq3
write.csv(wordFreq3,"top_trigram.csv")

#Trigram wordcloud
wordcloud(df3$term,df3$freq,min.freq=100,random.order = F,rot.per=0.3,colors=brewer.pal(8, "Dark2"))
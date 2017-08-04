#Import required libraries
library(NLP)
library(tm)
library(wordcloud)
library(RColorBrewer)

getwd()
setwd("C:\\xxxx")

flipkart <- read.csv("flipkart_train.csv",header = T,sep=',')
str(flipkart)

names(flipkart)
table(flipkart$Feedback)

Feedback <- flipkart[flipkart$Feedback==1,]
dim(Feedback)

#-------------------------Tag the sentiments---------------------------

tweets.text<-Feedback$text

#Read the dictionaries
pos = scan('positive-words.txt',what='character',comment.char=';')
neg = scan('negative-words.txt',what='character',comment.char=';')

#Adding some additional words to dictionaries
pos[2007]<-c("rocks")
neg[4784:4811]<-c("sick","still","reason","acceptin","fake","Ethics","poor","unpaid","chor","how","Flop","Fucking","thieves","holes","clueles","delayed",
                  "refund","fool","fooling","down","resolve","BoycottFlipkart","worstcustomercareplease","resolve","Poor","Shockingly","damaged","replacement")

#Famous Jeffreybreen Algorithm to "Tag" sentiments to sentences

score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  #we got a vector of sentences. plyr will handle a list
  #or a vector as an "l" for us
  #we want a simple array of scores back, so we use
  #"l" + "a" + "ply" = "laply":
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    #clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence) #removes punctuations
    sentence = gsub('[[:cntrl:]]', '', sentence) #removes control characters
    sentence = gsub('\\d+', '', sentence) #removes digits
    
    #and convert to lower case:
    sentence = tolower(sentence)
    
    #split sentences into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    
    #sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    #compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    #match() returns the position of the matched term or NA
    #we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    #and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

analysis<-score.sentiment(tweets.text, pos, neg, .progress="text")
names(analysis)
View(analysis)
str(analysis)

#Checking out overall sentiment
table(analysis$score)
mean(analysis$score)
hist(analysis$score)

analysis$text<-as.character(analysis$text)
str(analysis)
analysis$sentiment<-ifelse(analysis$score>0,"positive",
                           ifelse(analysis$score<0,"negative","neutral"))
table(analysis$sentiment)

#Cleaning the data again
analysis$text = gsub('[[:punct:]]', '', analysis$text)
str(analysis)
head(analysis,5)

write.csv(analysis,'feedback_classified.csv')
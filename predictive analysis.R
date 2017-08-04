#Consider manually clalssified sample (3547) & split into train and test before applying predictive modeling
#Call for the required libraries
library(NLP)
library(tm)
library(RTextTools)
library(caret)
library(e1071)
library(plyr)
library(car)

#set wordking directory
getwd()
setwd("C:\\Keyuri\\Gramener project")

#Import the data
flipkart <- read.csv("flipkart_train.csv",header = T,sep=',')
str(flipkart)
dim(flipkart)

#Split into train and test
sampling<-sort(sample(nrow(flipkart),nrow(flipkart)*.7))
length(sampling)
train<-flipkart[sampling,]
test<-flipkart[-sampling,]
#check number of rows & column names
nrow(train)
nrow(test)
names(train)
names(test)

#combine train and test data
tweets <- rbind(train,test)

#recode 0 to 2 and keep 1 unchanged
tweets$Deals.Offers <- recode(tweets$Deals.Offers,"0=2")
tweets$Products.Categories <- recode(tweets$Products.Categories,"0=2")
tweets$Events.Occasions <- recode(tweets$Events.Occasions,"0=2")
tweets$Competitors <- recode(tweets$Competitors,"0=2")
tweets$Contests <- recode(tweets$Contests,"0=2")
tweets$Flipkartbigsale <- recode(tweets$Flipkartbigsale,"0=2")
tweets$Marketing <- recode(tweets$Marketing,"0=2")
tweets$Others <- recode(tweets$Others,"0=2")
tweets$Feedback <- recode(tweets$Feedback,"0=2")

#building a text corpus and cleaning it
my_corpus<-VCorpus(VectorSource(tweets$text))
my_corpus<-tm_map(my_corpus,content_transformer(tolower))
my_corpus<-tm_map(my_corpus,removeNumbers)
removeURL <- function(x) gsub("(http[^ ]*)", "", x)
my_corpus<-tm_map(my_corpus,content_transformer(removeURL))
removeNumPunct<-function(x) gsub("[^[:alpha:][:space:]]*","",x)
my_corpus<-tm_map(my_corpus,content_transformer(removeNumPunct))
my_corpus<-tm_map(my_corpus, removeWords,stopwords("english"))
my_corpus<-tm_map(my_corpus,stripWhitespace)
my_corpus<-tm_map(my_corpus,stemDocument,language = "english")

#convert corpus to dataframe and then matrix
df <- data.frame(text = unlist(sapply(my_corpus,'[',"content")),stringsAsFactors=F)
#Remove sparse terms with 1% threshold
matrix <-create_matrix(df$text, removeSparseTerms = 0.99)
dim(matrix)
mat = as.matrix(matrix)
dim(mat)

#predictive modeling for deals/offers
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,19]
test.y <- tweets[2483:3547,19]

#Naive Bayes
model_deals <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_deals,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#Prediction using SVM,RF,TREE,MAXENT,BAGGING & ensemble of models for Deals/offers

#Build the data to specify response variable, training set, testing set
container = create_container(mat,(as.numeric(tweets[,19])), trainSize=1:2482, testSize=2483:3547,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_deals = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_deals)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:3547,19])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,19])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,19])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,19])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,19])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,19])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,19])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,19])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,19])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,19])), results[,"SVM_LABEL"])

#MODEL SUMMARY
analytics = create_analytics(container, results)
summary(analytics)


#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy #threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,19])), results1[,"majority"])

#predictive modeling for products/categories
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,20]
test.y <- tweets[2483:3547,20]

#Naive Bayes
set.seed(200)
model_prod<-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_prod,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble
#Build the data to specify response variable, training set, testing set.
container = create_container(mat,(as.numeric(tweets[,20])), trainSize=1:2482, testSize=2483:3547,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_prod = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_prod)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:3547,20])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,20])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,20])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,20])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,20])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,20])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,20])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,20])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,20])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,20])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]

#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,20])), results1[,"majority"])

#predictive modeling for competitors
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,22]
test.y <- tweets[2483:3547,22]

#Naive Bayes
set.seed(202)
model_comp<-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_comp,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat,(as.numeric(tweets[,22])), trainSize=1:2482, testSize=2483:3547,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_comp = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_comp)

# Accuracy Table
# Confusion Matrix

table(((tweets[2483:3547,22])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,22])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,22])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,22])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,22])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,22])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,22])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,22])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,22])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,22])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,22])), results1[,"majority"])

#Predictive modeling for flipkartbigsale
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,24]
test.y <- tweets[2483:3547,24]

#Naive Bayes algo
model_bigsale <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_bigsale,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat,(as.numeric(tweets[,24])), trainSize=1:2482,testSize=2483:3547,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_bigsale = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_bigsale)

# Accuracy Table
# Confusion Matrix

table(((tweets[2483:3547,24])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,24])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,24])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,24])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,24])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,24])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,24])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,24])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,24])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,24])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,24])), results1[,"majority"])

#predictive modeling for marketing
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,25]
test.y <- tweets[2483:3547,25]

#Naive Bayes algo
model_mk<-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_mk,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat,(as.numeric(tweets[,25])), trainSize=1:2482, testSize=2483:3547,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_mk = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_mk)

# Accuracy Table
# Confusion Matrix

table(((tweets[2483:3547,25])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,25])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,25])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,25])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,25])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,25])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,25])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,25])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,25])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,25])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,25])), results1[,"majority"])

#Predictive modeling for Feedback
train.x <- mat[1:2482,-1]
test.x <- mat[2483:3547,-1]
train.y <- tweets[1:2482,27]
test.y <- tweets[2483:3547,27]

#Naive Bayes algo
model_fb <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_fb,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat,(as.numeric(tweets[,27])), trainSize=1:2482, testSize=2483:3547,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_fb = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_fb)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:3547,27])), results[,"FORESTS_LABEL"])
table(((tweets[2483:3547,27])), results[,"MAXENTROPY_LABEL"])
table(((tweets[2483:3547,27])), results[,"TREE_LABEL"])
table(((tweets[2483:3547,27])), results[,"BAGGING_LABEL"])
table(((tweets[2483:3547,27])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,27])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,27])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,27])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,27])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets[2483:3547,27])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets[2483:3547,27])), results1[,"majority"])

#Out of time testing for the June tweets (sample=1500 & manually classifed for deals, products & feedback (other labels were absent))
#Import test data which also has classificatioon columns
flipkart_test <- read.csv("flipkart_test_classified.csv",header = T,sep=',')

str(flipkart_test)
dim(flipkart_test)

test1 <- flipkart_test
dim(test1)

tweets1 <- rbind(train,test1)
names(tweets)

#recode 1,0 to 1,2 (done for the predictive modeling..)
tweets1$Deals.Offers <- recode(tweets1$Deals.Offers,"0=2")
tweets1$Products.Categories <- recode(tweets1$Products.Categories,"0=2")
tweets1$Feedback <- recode(tweets1$Feedback,"0=2")

#tdm for test data
my_corpus1<-VCorpus(VectorSource(tweets1$text))
my_corpus1<-tm_map(my_corpus1,content_transformer(tolower))
my_corpus1<-tm_map(my_corpus1,removeNumbers)
removeURL <- function(x) gsub("(http[^ ]*)", "", x)
my_corpus1<-tm_map(my_corpus1,content_transformer(removeURL))
removeNumPunct<-function(x) gsub("[^[:alpha:][:space:]]*","",x)
my_corpus1<-tm_map(my_corpus1,content_transformer(removeNumPunct))
my_corpus1<-tm_map(my_corpus1, removeWords,stopwords("english"))
my_corpus1<-tm_map(my_corpus1,stripWhitespace)
my_corpus1<-tm_map(my_corpus1,stemDocument,language = "english")

#convert back corpus to dataframe and then matrix
df1 <- data.frame(text = unlist(sapply(my_corpus1,'[',"content")),stringsAsFactors = F)
dim(df1)
matrix1 <-create_matrix(df1$text, removeSparseTerms = 0.99)
dim(matrix1)
mat1 = as.matrix(matrix1)
mat1 <- as.data.frame(mat1)
dim(mat1)

#predictive modelingn for deals/offers
train.x <- mat1[1:2482,-1]
test.x <- mat1[2483:3982,-1]
train.y <- tweets1[1:2482,19]
test.y <- tweets1[2483:3982,19]

#Naive Bayes
model_deals <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_deals,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat1,(as.numeric(tweets1[,19])), trainSize=1:2482, testSize=2483:3982,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_deals = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_deals)

# Accuracy Table
# Confusion Matrix

table(((tweets1[2483:3982,19])), results[,"FORESTS_LABEL"])
table(((tweets1[2483:3982,19])), results[,"MAXENTROPY_LABEL"])
table(((tweets1[2483:3982,19])), results[,"TREE_LABEL"])
table(((tweets1[2483:3982,19])), results[,"BAGGING_LABEL"])
table(((tweets1[2483:3982,19])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results[,"SVM_LABEL"])

#MODEL SUMMARY
analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy
#threshold

analytics@ensemble_summary

#--------------------------------CROSS VALIDATION----------------------------
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets1[2483:3982,19])), results1[,"majority"])

#predictive modeling for products/categories
train.x <- mat1[1:2482,-1]
test.x <- mat1[2483:3982,-1]
train.y <- tweets1[1:2482,20]
test.y <- tweets1[2483:3982,20]

#Naive Bayes
model_prod <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_prod,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat1,(as.numeric(tweets1[,20])), trainSize=1:2482, testSize=2483:3982,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_deals = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_deals)

# Accuracy Table
# Confusion Matrix
table(((tweets1[2483:3982,20])), results[,"FORESTS_LABEL"])
table(((tweets1[2483:3982,20])), results[,"MAXENTROPY_LABEL"])
table(((tweets1[2483:3982,20])), results[,"TREE_LABEL"])
table(((tweets1[2483:3982,20])), results[,"BAGGING_LABEL"])
table(((tweets1[2483:3982,20])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(((tweets1[2483:3982,20])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,20])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,20])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,20])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,20])), results[,"SVM_LABEL"])

#MODEL SUMMARY

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary
#CROSS VALIDATION
N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets1[2483:3982,20])), results1[,"majority"])

#predictive modelingn for feedback
train.x <- mat1[1:2482,-1]
test.x <- mat1[2483:3982,-1]
train.y <- tweets1[1:2482,27]
test.y <- tweets1[2483:3982,27]

#Naive Bayes
model_prod <-naiveBayes(train.x,as.factor(train.y))

# Accuracy Table
# Confusion Matrix
results<-predict(model_prod,test.x)
table(pred=results,test.y)
confusionMatrix(results,test.y)

#ensemble algo
#Build the data to specify response variable, training set, testing set.
container = create_container(mat1,(as.numeric(tweets1[,27])), trainSize=1:2482, testSize=2483:3982,virgin=F)
#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_deals = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_deals)

# Accuracy Table
# Confusion Matrix
table(((tweets1[2483:3982,27])), results[,"FORESTS_LABEL"])
table(((tweets1[2483:3982,27])), results[,"MAXENTROPY_LABEL"])
table(((tweets1[2483:3982,27])), results[,"TREE_LABEL"])
table(((tweets1[2483:3982,27])), results[,"BAGGING_LABEL"])
table(((tweets1[2483:3982,27])), results[,"SVM_LABEL"])

results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
results$MAXENTROPY_LABEL <- as.numeric(results$MAXENTROPY_LABEL)
results$TREE_LABEL <- as.numeric(results$TREE_LABEL)
results$BAGGING_LABEL <- as.numeric(results$BAGGING_LABEL)
results$SVM_LABEL <- as.numeric(results$SVM_LABEL)

# recall accuracy
recall_accuracy(((tweets1[2483:3982,27])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,27])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,27])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,27])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric((tweets1[2483:3982,27])), results[,"SVM_LABEL"])

#MODEL SUMMARY
analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#ENSEMBLE AGREEMENT
#Coverage: simply refers to the percentage of documents that meet the recall accuracy 
#threshold

analytics@ensemble_summary

#--------------------------------CROSS VALIDATION----------------------------
N=4
set.seed(2714)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

results1<-results[,c(1,3,5,7,9)]
head(results1)
#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}
results1$majority<-as.numeric(results1$majority)

# recall accuracy
recall_accuracy(as.numeric((tweets1[2483:3982,27])), results1[,"majority"])


#predicting for the unclassified tweets (n=1520)
flipkart_test <- read.csv("flipkart_unclassified.csv",header = T,sep=',')
str(flipkart_test)
train_1 <- test
test<-flipkart_test
nrow(train)
nrow(test)
names(train)
names(test)

test$Deals.Offers <- 2
test$Products.Categories <- 2
test$Events.Occasions <- 2
test$Competitors <- 2
test$Contests <- 2
test$Flipkartbigsale <- 2
test$Marketing <- 2
test$Others <- 2
test$Feedback <- 2
test$Merger <- 2
dim(test)
dim(train)

names(train)
names(test)
tweets <- rbind(train,test)
dim(tweets)
names(tweets)

#recode 0 to 2 and keep 1 unchanged
tweets$Deals.Offers <- recode(tweets$Deals.Offers,"0=2")
tweets$Products.Categories <- recode(tweets$Products.Categories,"0=2")
tweets$Events.Occasions <- recode(tweets$Events.Occasions,"0=2")
tweets$Competitors <- recode(tweets$Competitors,"0=2")
tweets$Contests <- recode(tweets$Contests,"0=2")
tweets$Flipkartbigsale <- recode(tweets$Flipkartbigsale,"0=2")
tweets$Marketing <- recode(tweets$Marketing,"0=2")
tweets$Others <- recode(tweets$Others,"0=2")
tweets$Feedback <- recode(tweets$Feedback,"0=2")


#building a text corpus
#Source for the corpus
my_corpus<-VCorpus(VectorSource(tweets$text))
my_corpus<-tm_map(my_corpus,content_transformer(tolower))
my_corpus<-tm_map(my_corpus,removeNumbers)
removeURL <- function(x) gsub("(http[^ ]*)", "", x)
my_corpus<-tm_map(my_corpus,content_transformer(removeURL))
#remove anything other than English letters or space
removeNumPunct<-function(x) gsub("[^[:alpha:][:space:]]*","",x)
my_corpus<-tm_map(my_corpus,content_transformer(removeNumPunct))
my_corpus<-tm_map(my_corpus, removeWords,stopwords("english"))
my_corpus<-tm_map(my_corpus,stripWhitespace)
#Stem words in corpus
my_corpus<-tm_map(my_corpus,stemDocument,language = "english")
#convert back corpus to dataframe and then matrix
df <- data.frame(text = unlist(sapply(my_corpus,'[',"content")),stringsAsFactors = F)
matrix <-create_matrix(df$text, removeSparseTerms = 0.99)
dim(matrix)
require(car)
mat <- as.matrix(matrix)
dim(mat)
names(tweets)
dim(tweets)

#random forest
container = create_container(mat,(as.numeric(tweets[,19])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_deals = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_deals)

table(((tweets[2483:4003,19])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Deals.Offers <- results$FORESTS_LABEL
table(tweets[2483:4003,19],results$FORESTS_LABEL)


#predicting for products/categories
container = create_container(mat,(as.numeric(tweets[,20])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_prod = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_prod)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:4003,20])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Products.Categories <- results$FORESTS_LABEL

#predicting for competitor
container = create_container(mat,(as.numeric(tweets[,22])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_comp = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_comp)

# Accuracy Table
# Confusion Matrix

table(((tweets[2483:4003,22])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Competitors <- results$FORESTS_LABEL

#predicting for flipkartbigsale
container = create_container(mat,(as.numeric(tweets[,24])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_bigsale = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_bigsale)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:4003,24])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Flipkartbigsale <- results$FORESTS_LABEL

#predicting for Marketing
container = create_container(mat,(as.numeric(tweets[,25])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_mk = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_mk)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:4003,25])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Marketing <- results$FORESTS_LABEL

#predicting for the feedback tweets
container = create_container(mat,(as.numeric(tweets[,27])), trainSize=1:2482, testSize=2483:4003,virgin=F)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models_fb = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models_fb)

# Accuracy Table
# Confusion Matrix
table(((tweets[2483:4003,27])), results[,"FORESTS_LABEL"])
results$FORESTS_LABEL <- as.numeric(results$FORESTS_LABEL)
test$Feedback <- results$FORESTS_LABEL

table(results$FORESTS_LABEL)

output <- rbind(train,train_1,test)
dim(output)
write.csv(output,"output_predicted.csv")
table(output$Deals.Offers)

#recode 0 to 2 and keep 1 unchanged
output$Deals.Offers <- recode(output$Deals.Offers,"0=2")
output$Products.Categories <- recode(output$Products.Categories,"0=2")
output$Events.Occasions <- recode(output$Events.Occasions,"0=2")
output$Competitors <- recode(output$Competitors,"0=2")
output$Contests <- recode(output$Contests,"0=2")
output$Flipkartbigsale <- recode(output$Flipkartbigsale,"0=2")
output$Marketing <- recode(output$Marketing,"0=2")
output$Others <- recode(output$Others,"0=2")
output$Feedback <- recode(output$Feedback,"0=2")

round(prop.table(table(output$Deals.Offers))*100,0)
round(prop.table(table(output$Products.Categories))*100,0)
round(prop.table(table(output$Competitors))*100,0)
round(prop.table(table(output$Flipkartbigsale))*100,0)
round(prop.table(table(output$Marketing))*100,0)
round(prop.table(table(output$Feedback))*100,0)

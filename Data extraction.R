#Set working directory
setwd("C:\xxxxx")

#install.packages(c("devtools", "rjson", "bit64"))
#RESTART R session!
library(devtools)
library(twitteR)

#Now the twitteR package is up-to-date 
#setup_twitter_oauth() function uses the httr package
#Twitter Authentication with R

Consumer_key<- "xxxxxxxxxxxxx"
Consumer_secret <- "xxxxxxxxxxxxx"
access_token <- "xxxxxxxxxxxxx"
access_token_secret <- "xxxxxxxxxxxxx"
setup_twitter_oauth(Consumer_key,Consumer_secret,access_token,access_token_secret)

Yes

#You can Search
gst_tweet <- searchTwitter("#flipkart",n=10000)
#make data frame
gst_tweet <- do.call("rbind", lapply(gst_tweet, as.data.frame))
#write to csv file (or your RODBC code)
write.csv(gst_tweet,file="flipkart.csv")


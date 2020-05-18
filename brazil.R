rm(list=ls())
cat("\014")
graphics.off()
#install.packages("glmnet")
#install.packages("ggplot2")
#install.packages("gridExtra")
#install.packages("randomForest")
#library(glmnet)
#library(ggplot2)
#library(gridExtra)
#library(randomForest)

#1. Data Cleaning
br= read.csv("brazil.csv")
sum(is.na(br)) #41085
summary(br)

#change AREA data type to numeric
br$AREA = as.numeric(gsub(",","",br$AREA,fixed=TRUE))#remove commas format

#remove columns that contain categorical values and too many missing values
na = c(1,2,3,24,25,30,31,33,43,44,65,67,68,78,69,70,71,72,73,74,79,80)
br1= br[c(1:4000),-na] 
sum(is.na(br1)) #260 NAss
br1[is.na(br1)] = 0 #set the rest of the NAs equal to 0


#2.Create X matrix and vector y 
X = data.matrix(br1[,-34])#GDP_CAPITA is col 34, our y
y = br1[,34]
X.orig   =   X
p = dim(br1)[2] - 1 #58 x variables
n = dim(br1)[1] #4000 observations

#standardize data points so that their means are close to 0 and sd = 1
X = scale(X)
apply(X,2,'mean')
apply(X,2,'sd')

#Randomly split the data, train = 80% , test = 20%
n.train = floor(n*0.8)
n.test = n-n.train

#M is number of times loop run
M = 5

#to store R-squared of train and test sets using Lasso
Rsq.train.ls = rep(0,M)
Rsq.test.ls = rep(0,M)
#to store R-squared of train and test sets using Elastic Net
Rsq.train.en = rep(0,M)
Rsq.test.en = rep(0,M)
#to store R-squared of train and test sets using Random Forest
Rsq.train.rf = rep(0,M)
Rsq.test.rf= rep(0,M)
#to store R-squared of train and test sets using Ridge
Rsq.train.rd = rep(0,M)
Rsq.test.rd= rep(0,M)

#store residuals of train and test sets for each method when running loop 1 time
train.res.rd = rep(0,n.train)
test.res.rd = rep(0,n.test)
train.res.en = rep(0,n.train)
test.res.en = rep(0,n.test)
train.res.ls = rep(0,n.train)
test.res.ls = rep(0,n.test)
train.res.rf = rep(0,n.train)
test.res.rf = rep(0,n.test)

set.seed(1)
#3.Calculate R-squares for Ridge Regression, Lasso, Elastic Net, Random Forest models
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
start_time_rd = Sys.time()
a = 0 #ridge
  cv.fit.rd        =     cv.glmnet(X.train,y.train,alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.rd$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.rd[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rd[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
 #loop run one out of M times
  if (m<2){
    train.res.rd        =     as.vector(y.train - y.train.hat)
    test.res.rd         =     as.vector(y.test - y.test.hat)
    boxplot(train.res.rd,test.res.rd, horizontal = TRUE, at= c(1,2), 
          names=c("Train","Test"), main = "Ridge Regression")}
end_time_rd = Sys.time()
time_rd = end_time_rd - start_time_rd

start_time_en = Sys.time()
a=0.5# elastic-net
   cv.fit.en        =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
   fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.en$lambda.min)
   y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
   y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
   Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
   Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
   #loop run one out of M times
   if(m<2){
       train.res.en     =     y.train - as.vector(y.train.hat)
       test.res.en      =     y.test - as.vector(y.test.hat)
       boxplot(train.res.en,test.res.en, horizontal = TRUE,at= c(1,2), 
           names=c("Train","Test"), main = "Elastic Net") }
end_time_en = Sys.time()
time_en = end_time_en - start_time_en

start_time_ls = Sys.time()
a=1# lasso
   cv.fit.ls        =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
   fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit.ls$lambda.min)
   y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
   y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
   Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
   Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
   #loop run one out of M times
   if(m<2){
      train.res.ls     =     y.train - as.vector(y.train.hat)
      test.res.ls      =     y.test - as.vector(y.test.hat)
      boxplot(train.res.ls,test.res.ls, horizontal = TRUE,at= c(1,2),names=c("Train","Test"), main = "Lasso")}
end_time_ls = Sys.time()
time_ls = end_time_ls - start_time_ls

start_time_rf = Sys.time()  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  #only when M = 1
  if(m<2){
    train.res.rf        =     y.train - as.vector(y.train.hat)
    test.res.rf         =     y.test  - as.vector(y.test.hat)
    boxplot(train.res.rf, test.res.rf,horizontal = TRUE,at= c(1,2), 
          names=c("Train","Test"), main = "Random Forest")}
end_time_rf = Sys.time()
time_rf = end_time_rf - start_time_rf
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f,  
              Rsq.test.ls=%.2f, Rsq.test.rd=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f, 
              Rsq.train.ls=%.2f, Rsq.train.rd=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m], 
              Rsq.test.ls[m],  Rsq.test.rd[m],  Rsq.train.rf[m], Rsq.train.en[m], 
              Rsq.train.ls[m], Rsq.train.rd[m]))
}

#time elapsed by each model
data.frame(time_rd,time_en,time_ls,time_rf)

#Boxplots of the test and train R-squares
boxplot(Rsq.test.rf,Rsq.test.en,Rsq.test.ls,Rsq.test.rd, main = "Test R-square",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RF","EN","LS","RD"),
        col = "orange",horizontal = FALSE)


boxplot(Rsq.train.rf,Rsq.train.en,Rsq.train.ls,Rsq.train.rd, main = "Train R-square",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RF","EN","LS","RD"),
        col = "light blue",horizontal = FALSE)

# Plot 10 fold cross validation curves
plot(cv.fit.en,sub = "Elastic Net", cex.sub = 1) #elasticnet
plot(cv.fit.ls,sub = "Lasso", cex.sub = 1) #lasso
plot(cv.fit.rd,sub = "Ridge", cex.sub = 1) #ridge

#4.Bootstrap 
#Create bootstraped samples
bootstrapSamples =     5
#Store the importance of each coefficient in RF and the betas in LS,EN,RD
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.ls.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)   
beta.rd.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)   

start_time_bs = Sys.time()
for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en  
  a                =     0.5
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  #fit bs ls
  a                =      1
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ls.bs[,m]   =     as.vector(fit$beta)
  #fit bs rd
  a                =     0
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rd.bs[,m]   =     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
end_time_bs = Sys.time()
end_time_bs - start_time_bs #total time elapsed for bootstrap samples

# calculate bootstrapped standard errors 
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ls.bs.sd    = apply(beta.ls.bs, 1, "sd")
rd.bs.sd    = apply(beta.rd.bs, 1, "sd")


# fit rf to the whole data
rf.whole         =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en,rd,ls to the whole data
a=0.5 # elastic-net
cv.en            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.en$lambda.min)
a=0 # ridge
cv.rd            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rd           =     glmnet(X, y, alpha = a, lambda = cv.rd$lambda.min)
a=1 #lasso
cv.ls            =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.ls           =     glmnet(X, y, alpha = a, lambda = cv.ls$lambda.min)

#store the importance of the coefficients with its error (2sd) in a data frame
betaS.rf               =     data.frame(c(1:p), as.vector(rf.whole$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")
#store betas in en,rd, ls with its error (2sd) in a data frame
betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.rd               =     data.frame(c(1:p), as.vector(fit.rd$beta), 2*rd.bs.sd)
colnames(betaS.rd)     =     c( "feature", "value", "err")

betaS.ls               =     data.frame(c(1:p), as.vector(fit.ls$beta), 2*ls.bs.sd)
colnames(betaS.ls)     =     c( "feature", "value", "err")

#barplots with bootstrapped error bars for rf, en
rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Random Forest")


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Elastic Net")

grid.arrange(rfPlot, enPlot, nrow = 2)

#barplots with bootstrapped error bars for rd, ls
rdPlot =  ggplot(betaS.rd, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Ridge")

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Lasso")
grid.arrange(rdPlot, lsPlot, nrow = 2)

#rearrange the order of betas according to the order of the importance of betas in rf
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rd$feature     =  factor(betaS.rd$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Random Forest")

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle("Elastic Net")

grid.arrange(rfPlot, enPlot, nrow = 2)

rdPlot =  ggplot(betaS.rd, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Ridge")

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+ ggtitle("Lasso")

grid.arrange(rdPlot, lsPlot, nrow = 2)



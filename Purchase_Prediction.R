# Author  	: Vamshi Krishna Indla
# Email		: vamshi.indla@gmail.com
# File created :  21-Nov-15 
###################################################################################

#########################################
# Set up Environment
#########################################
dev.off()
rm(list=ls(all.names=TRUE))
set.seed(1234)

#   Enable Multi-Core Processing
library(doParallel);
cluster <- makeCluster(detectCores());
registerDoParallel(cluster);

#########################################
# Install Libraries
#########################################
library(Metrics) 
library(readr)
library(plyr)

#########################################
# Set the Directory and read the train data into a dataframe named train
#########################################
setwd("E:\\Certifications\\ISB CBA\\00 Kaggle\\Hack3-ecommerce/")
data = read.csv("train.csv",header=T)
test = read.csv("test.csv",header=T)

#########################################
# Feature Engineering
#########################################
data$Marital_Status <- as.factor(data$Marital_Status)
data$Occupation <- as.factor(data$Occupation)
data$Product_Category_2[is.na(data$Product_Category_2)] <- 999 
data$Product_Category_3[is.na(data$Product_Category_3)] <- 999 
data$Product_Category_1 <- as.factor(data$Product_Category_1) 
data$Product_Category_2 <- as.factor(data$Product_Category_2) 
data$Product_Category_3 <- as.factor(data$Product_Category_3) 

test$Marital_Status <- as.factor(test$Marital_Status)
test$Occupation <- as.factor(test$Occupation)
test$Product_Category_2[is.na(test$Product_Category_2)] <- 999 
test$Product_Category_3[is.na(test$Product_Category_3)] <- 999 
test$Product_Category_1 <- as.factor(test$Product_Category_1) 
test$Product_Category_2 <- as.factor(test$Product_Category_2) 
test$Product_Category_3 <- as.factor(test$Product_Category_3) 
 
#####################################################
# H2O Deep Learning 
#####################################################
library(h2o)
# Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='1G')

## Set up variable to use all features other than those specified here
features<-colnames(data[,-c(1,2,12)])
trainHex <- as.h2o(data[,-c(1,2)])
validHex <- as.h2o(test[,-c(1,2)])

dlHex <- h2o.deeplearning(x=features,
                          y="Purchase", 
                          training_frame=trainHex)


model2_testpredict <-as.data.frame(h2o.predict(dlHex,validHex))

model2_predictions<-as.data.frame(h2o.predict(dlHex,trainHex))
model2 <- rmse(model2_predictions,data$Purchase)
model2 
rm(dlHex)
#####################################################
# H2O gbm
#####################################################
gbmHex <- h2o.gbm(x=features,
                  y="Purchase", 
                  training_frame=trainHex)

model3_testpredict<-as.data.frame(h2o.predict(gbmHex,validHex))

model3_predictions<-as.data.frame(h2o.predict(gbmHex,trainHex))
model3 <- rmse(model3_predictions,data$Purchase)
model3
rm(gbmHex)
#####################################################
# Linear Regression
#####################################################
library(MASS)
lmmodel <- lm(sqrt(Purchase)~.,data=data[,-c(1,2)])
model4_testpredict <- predict(lmmodel, type="response", test)
model4_testpredict <- model4_testpredict^2

model4_predictions <- predict(lmmodel, type="response", data)
model4_predictions <- (model4_predictions)^2
model4 <- rmse(model4_predictions,data$Purchase)
model4
rm(lmmodel)

########################################
# XGBoost Attempt
########################################
library(xgboost)
xtraining <- model.matrix(~.,data=data[, -c(1:2,12)])
xtest <- model.matrix(~.,data=test[, -c(1:2,12)])

dval<-xgb.DMatrix(data=xtraining,label=data$Purchase)
dtrain<-xgb.DMatrix(data=xtraining,label=data$Purchase)
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                #booster = "gblinear",
                eta                 = 0.1, 
                max_depth           = 8, 
                subsample           = 0.7, 
                colsample_bytree    = 0.7 
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 700, 
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    metrics             = "rmse"
)

model5_testpredict <- predict(clf, xtest) 

model5_predictions<- predict(clf, xtraining) 
model5 <- rmse(model5_predictions,data$Purchase)
model5
rm(clf)
########################################
# Merge Predictions
########################################
weight2 <- 1 - (model2/ sum(model2,model3,model4,model5))
weight3 <- 1 - (model3/ sum(model2,model3,model4,model5))
weight4 <- 1 - (model4/ sum(model2,model3,model4,model5))
weight5 <- 1 - (model5/ sum(model2,model3,model4,model5)) 

weights2 <- (weight2/ sum(weight2,weight3,weight4,weight5))
weights3 <- (weight3/ sum(weight2,weight3,weight4,weight5))
weights4 <- (weight4/ sum(weight2,weight3,weight4,weight5))
weights5 <- (weight5/ sum(weight2,weight3,weight4,weight5)) 

model_predictions <- ( weights2*model2_predictions ) + ( weights3*model3_predictions ) + 
  ( weights4*model4_predictions ) + ( weights5*model5_predictions)

modelx<- rmse(model_predictions,data$Purchase)
dval<-xgb.DMatrix(data=xtraining,label=data$Purchase)
dtrain<-xgb.DMatrix(data=xtraining,label=data$Purchase)
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                #booster = "gblinear",
                eta                 = 0.05, 
                max_depth           = 10, 
                subsample           = 1.0, 
                colsample_bytree    = 0.3, 
                gamma 		    = 0.0
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1500, 
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    metrics             = "rmse"
)

model5_testpredict <- predict(clf, xtest) 

model5_predictions<- predict(clf, xtraining) 
model5 <- rmse(model5_predictions,data$Purchase)
model5

#####################################################
# Test Prediction
#####################################################
model_testpredict <- ( weights2*model2_testpredict ) + ( weights3*model3_testpredict ) + 
  ( weights4*model4_testpredict ) + ( weights5*model5_testpredict )

# save csv file for submission
submit <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = model_testpredict)
colnames(submit) <- c("User_ID","Product_ID","Purchase")
write.csv(submit, file = "Xgboost.csv", row.names = FALSE)


save(list=ls(), file=paste0("Ensemble", ".RData"))

# Another program to run the models
# load("validation.RData", verbose=TRUE)

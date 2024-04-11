#ADDING ALL OF THE LIBRARY NEEDED FOR THIS PROGRAM
library(corrplot)
library(caret)
library(Metrics)
library(neuralnet)
library(nnet)
library(psych)
library(randomForest)

##################################################################################

#Setting working directory
setwd("C:/Users/jdvil/OneDrive/MIS 790/Comp Exam")
#SETTING DATA AS RAW
df<-(read.csv("data_PM10.csv"))

##################################################################################

#REVIEWING COLUMN NAMES
colnames(df)
#RENAMING COLUMNS
names(df)[c(2,15,22,23,26,28,31,33,35,36,37,19)] <- c("ToL","ME_Fuel","LF_ME_Maneuver",
                                        "T_min_Hotel_Format","Energy_ME_Maneuver",
                                        "Energy_AE_Maneuver","ME_Maneuver",
                                        "AE_Maneuver","Boiler_Maneuver",
                                        "Boiler_Hoteling","Subtotal_NoBoiler","LF_ME_Cruising")

##################################################################################

#EXECUTING RENAMING OF COLUMNS
colnames(df)
#REVIEWING THE DATA TYPE OF EACH VARIABLE
str(df)

##################################################################################

#DROPPING IRREVELATE COLUMNS
df <- subset(df,select=-c(Motor_Ship,Docking_Day,Docking_Time,Sailing_Day,Departure_Time,
                          Permanence,Motor_Ship.1,IMO,Year_of_Construct,T_min_Hotel_Format,
                          A_ME,ToL,ME_Fuel,AE_Fuel,Engine_Speed,T_min_Hotell,Subtotal_NoBoiler))

#REORDERING THE COLUMNS IN THE DATASET
colnames(df[])
new_order <- c("Max_Speed","Avg_Speed","Gross_tonnage","Dead_Weight","Power_AE",
               "Power_ME","AE_Cruising","AE_Maneuver","AE_Hotelling",
               "ME_Cruising","ME_Maneuver","Boiler_Hoteling","Boiler_Maneuver",
               "LF_ME_Cruising","LF_ME_Maneuver","Energy_AE_Cruising",
               "Energy_AE_Hotelling","Energy_AE_Maneuver","Energy_ME_Cruising",
               "Energy_ME_Maneuver","Total")
df <- df[,new_order]

##################################################################################

#DISPLAYING NUMBER OF MISSING VALUES
sum(is.na(df))
#REMOVING MISSING VALUES' ROWS
df<-na.omit(df)
#DISPLAYING NUMBER OF ROWS
nrow(df)
#DOUBLE CHECKING MISSING VALUES
sum(is.na(df))

##################################################################################

#SPLIT DATA INTO TRAINING AND TESTING
set.seed(619)
TrainingIndex <- createDataPartition(df$Total, p = 0.75, list = FALSE)
TrainingSet <- df[TrainingIndex, ]
TestingSet <- df[-TrainingIndex, ]

#REASSIGNING TRAINING AND TESTING DATASETS
train.df <- TrainingSet
test.df <- TestingSet

##################################################################################

#CREATING CORRELATION PLOT
corrplot(cor(train.df),
         title="Correlation with Total PM10 Emissions",
         method = "circle",
         type = "upper",
         addCoef.col = "black",
         number.cex = 0.7,
         diag = T,
         tl.col = "black",
         mar = c(0,0,1,0),
         tl.srt = 45,
         tl.cex = 0.8)

##################################################################################

#REMOVE HIGHLY CORRELATED VARIABLES
highly_correlated <- findCorrelation(cor(train.df[, -which(names(train.df) == "Total")]),cutoff = .85)
train <- train.df[,-highly_correlated]
test <- train.df[,-highly_correlated]

##################################################################################

#DISPLAYING BOXPLOT OF VARS
boxplot(train.df,
        main="Boxplot of Training Data",
        xlab="Variables",
        col="green",
        pch=19,
        tl.srt=45,
        outline=TRUE)

#DISPLAYING BOXPLOT OF SCALED VARS
boxplot(scale(train.df),
        main="Boxplot of Normalized Training Data",
        xlab="Variables",
        col="green",
        pch=19,
        outline=TRUE)

#DISPLAYING CORRELATION PLOT OF ALL VARIABLES
corrplot(cor(train.df),title = "CorrPlot without Highly Correlated Variables",
         type="lower",
         diag = T,
         addCoef.col = "black",
         number.cex = 0.7,
         tl.col="black",
         mar = c(0,0,1,0),
         tl.srt=45)



#ASSIGNING RESAMPLING METHOD
ctrl <- trainControl(method = "cv",number=10)

##################################################################################

#BUILDING THE LM MODEL
set.seed(619)
lm.model <- train(Total ~ .,
             data = scale(train),
             method = "lm",
             trControl = ctrl)
#DISPLAYING THE LM MODEL RESULTS
lm.model

#REVIEWING THE VARIABLE IMPORTANCE
lm.importance <- varImp(lm.model)
varImp(lm.model)
plot(lm.importance)

#TESTING THE LM MODEL
pred.lm <- predict(lm.model, newdata = scale(test))

#COMPARING THE LM PREDICTIONS TO ACTUALS
plot(scale(test$Total), pred.lm,
     main = "Actual vs. LM Predicted",
     xlab = "Actual",
     ylab = "LM Predicted",
     pch=19,
     col=c("green","blue"))
abline(0, 1, col = "red")

##################################################################################

#BUILDING THE RANDOM FOREST MODEL
set.seed(619)
rf <- train(Total ~ .,
            data = train,
            method = "rf",
            trControl = ctrl)
#DISPLAYING THE RANDOM FOREST RESULTS
rf
#TESTING THE RF MODEL
pred.rf <- predict(rf, newdata = test)

#REVIEWING THE VARIABLE IMPORTANCE
importance <- varImp(rf)
varImp(rf)
plot(importance)

#COMPARING THE RF MODELS PREDICTIONS VS ACTUALS
plot(test$Total, pred.rf,
     main = "Actual vs. RF Predicted",
     xlab = "Actual",
     ylab = "RF Predicted",
     pch=19,
     col=c("green","blue"))
abline(0, 1, col = "red")

##################################################################################

#BUILDING THE KNN MODEL
set.seed(619)
knn <- train(Total ~ .
             data = scale(train),
             method = "knn",
             trControl = ctrl,
             tuneLength=20)
#DISPLAYING THE KNN RESULTS
knn
plot(knn)
#TESTING THE KNN MODEL
pred.knn <- predict(knn, newdata = scale(test))

#COMPARING THE KNN MODELS PREDICTIONS VS ACTUALS
plot(scale(test$Total), pred.knn,
     main = "Actual vs. KNN Predicted",
     xlab = "Actual",
     ylab = "KNN Predicted",
     pch=19,
     col=c("green","blue"))
abline(0,1, col = "red")

##################################################################################

#BUILDING THE NEURAL NETWORK MODEL
set.seed(619)
nnet <- neuralnet(Total~.,
                  data=scale(train),
                  hidden=5,
                  rep = TRUE,
                  startweights = NULL)
#MODEL'S RESULTS MATRIX
nnet
plot(nnet)
#TESTING THE MODEL
pred.nnet <- predict(nnet, newdata = scale(test))

#test <- as.data.frame(test)
#REVIEWING THE WEIGHT OF THE MODEL
weights <- nnet$weights
print(weights)

#CALCULATING THE METRICS
pred.nnet_unscaled <- pred.nnet * sd(train$Total) + mean(train$Total)
mae <- mean(abs(pred.nnet_unscaled - test$Total))
mse <- mean((pred.nnet_unscaled - test$Total)^2)
#PRINT RESULTS
print(paste("Mean Absolute Error (MAE):", mae))
print(paste("Mean Squared Error (MSE):", mse))

#COMPARING THE NEURAL NETWORK PREDICTIONS TO THE ACTUALS
plot(scale(test$Total), pred.nnet,
     main = "Actual vs. NNET Predicted",
     xlab = "Actual",
     ylab = "NNET Predicted",
     pch=19,
     col=c("green","blue"))
abline(0,1, col = "red")

##################################################################################

#CREATING LISTS TO COMPARE ALL MODELS' METRICS
model_names <- c("lm", "rf","knn", "nnet")
predictions_names <- c("pred.lm", "pred.rf","pred.knn", "pred.nnet")
mse <- rmse <- mae <- r_squared <- numeric(length(model_names))

#CALCULATING METRICS FOR EACH MODEL
for (i in seq_along(model_names)) {
  # Get the predictions for the current model
  predictions <- get(predictions_names[i])
  
  # Calculate MSE
  mse[i] <- mean((test$Total - predictions)^2)
  
  # Calculate RMSE
  rmse[i] <- sqrt(mse[i])
  
  # Calculate MAE
  mae[i] <- mean(abs(test$Total - predictions))
  
  # Calculate R-squared
  r_squared[i] <- cor(test$Total, predictions)^2
}
# COMBINE METRICS INTO A DF
metrics_df <- data.frame(
  Model = model_names,
  MSE = mse,
  RMSE = rmse,
  MAE = mae,
  R_squared = r_squared)

#SELECTING THE BEST PERFORMING MODEL BASED ON RMSE
best_model <- metrics_df[which.min(metrics_df$RMSE), ]
#PRINT THE COMPARISON TABLE
print(metrics_df)
#PRINT THE BEST MODEL
print("Best performing model:")
print(best_model)


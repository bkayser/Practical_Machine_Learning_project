library(caret)
library(dplyr)
setwd("~/courses/Practical_Machine_Learning/project")
training <- read.csv('./pml-training.csv', na.strings = c('','NA',"#DIV/0!"))

# Remove NA columns
missing_rowcount <- apply(training, MARGIN = 2, FUN=function(v) { sum(is.na(v)) })
table(missing_rowcount)
empty_vars <- which(missing_rowcount > 10000)

features <- select(training,
                  -classe,
                  -empty_vars,
                  -starts_with('raw_timestamp'),
                  -new_window,
                  -num_window,
                  -X, 
                  -user_name, 
                  -cvtd_timestamp)
# 'new_window' has little variance    
nzv <- nearZeroVar(features)

pp <- preProcess(features, method=c("center", "scale", "pca"), thresh=0.95)
features_pp <- predict(pp, features)

ctrl = trainControl(method='cv', 
                    number=10,
                    repeats=1)

model <- train(features_pp, training$classe, control=ctrl, method="rf" )

# Save the data
save(training, model, file="model.RData")

confusionMatrix(model)


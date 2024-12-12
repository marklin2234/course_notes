# SETUP

education_train = read.csv('module_Education_train_set.csv')
household_train = read.csv('module_HouseholdInfo_train_set.csv')
poverty_train = read.csv('module_SubjectivePoverty_train_set.csv')
education_test = read.csv('module_Education_test_set.csv')
household_test = read.csv('module_HouseholdInfo_test_set.csv')

library(tidyr)
library(dplyr)

poverty_train = poverty_train %>%
  separate(psu_hh_idcode, into=c("psu","hh","idcode"),sep="_",convert=TRUE) %>%
  rowwise() %>%
  mutate(
    poverty = which(c_across(starts_with("subjective_poverty_")) == 1)
  ) %>%
  ungroup() %>%
  select(psu,hh,idcode,poverty)

colnames(education_train) <- gsub("^q", "Q", colnames(education_train))

### What strategies should we use?
### 1. Lasso on each training set individually -> combine variates -> regress
### 2. Lasso on combined training set -> regress
### 3. How do we regress on variates? Elastic net, ridge regression, logit,
### least absolute error, robust ridge, select best model using APSE, or APAE
### Can try using a random forest

# Setup
## We want to remove some unecessary data.
## Household data Q4 and Q5 is DOB and age, q12 is just idcode, q17 is just
## idcode

## In education data, if q3 is NO (2) then q4 should be 0, not the average

train_data <- merge(household_train, education_train, by=c("psu","hh","idcode"),
                 all=FALSE)
train_data <- merge(train_data, poverty_train, by=c("psu","hh","idcode"),
                    all=FALSE)
test_data <- merge(household_test, education_test, by=c("psu","hh","idcode"))

## DATA PREPROCESSING

cols_to_replace <- c("Q04", "Q05", "Q06", "Q07", "Q08","Q12","Q13")
train_data[cols_to_replace] <- lapply(train_data[cols_to_replace], function(x) {
  x[is.na(x)] <- 0
  return(x)
})

train <- train_data %>% dplyr::sample_frac(0.8)
val <- dplyr::anti_join(train_data, train, by=c("psu","hh","idcode"))

x_train <- train %>% select(-poverty,-psu,-hh,-idcode,-hhid)
y_train <- train$poverty
x_val <- val %>% select(-poverty,-psu,-hh,-idcode)
y_val <- val$poverty

x_train[, sapply(x_train, is.numeric)] <- 
  lapply(x_train[, sapply(x_train, is.numeric)], 
         function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
x_train <- x_train[, !apply(is.na(x_train), 2, all)]

# Variable Selection

library(glmnet)
library(randomForest)

poly_x <- as.data.frame(lapply(x_train, function(col) {
  n_unique <- length(unique(col))
  if(n_unique > 2) {
    return(poly(col, degree = 2))
  } else {
    return(col) # Return original column if not enough unique values
  }
}))
CV = cv.glmnet(as.matrix(poly_x),y_train,alpha=1,standardize=TRUE,
               intercept=TRUE)

lasso_fit <- glmnet(as.matrix(poly_x), y_train,alpha=1,lambda=CV$min)
selected_vars <- which(coef(CV,s="lambda.min")[-1] != 0)

X_selected <- poly_x[,selected_vars]

rf <- randomForest(X_selected, factor(y_train), ntree=500, importance=TRUE)

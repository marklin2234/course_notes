# SETUP
set.seed(123)
library(dplyr)
library(tidyr)
library(MASS)
data <- read.csv("Data_Group3-Modified.csv", header=TRUE)
data <- data[!duplicated(data$X), ]
data <- data[-1,]

## Impute value

impute_previous_5_years <- function(data, year, column) {
  # Get the rows from the previous 5 years
  prev_years <- data %>% filter(X < year) %>% arrange(desc(X)) %>% head(5)
  
  # Calculate the mean of the previous 5 years
  avg_value <- mean(prev_years[[column]], na.rm = TRUE)
  
  return(avg_value)
}

new_row <- data.frame(
  Season = NA,
  Player = NA,
  X = 2005,
  G = impute_previous_5_years(data, 2005, "G"),
  GP = NA,
  G.GP = NA,
  Leage.average.goals.team.game = NA,
  Reg.Season.games = impute_previous_5_years(data, 2005, "Reg.Season.games"),
  X.1 = NA
)

data <- bind_rows(data, new_row) %>% arrange(X)

## Normalize data

data <- data$G / data$Reg.Season.games

data.ts <- ts(data, start=1918, end=2024,frequency=1)

plot(data.ts)
acf(data.ts)

y=as.vector(data.ts) 

boxcox.model = boxcox(lm(y~1))
optimal.lambda = boxcox.model$x[which.max(boxcox.model$y)] 
transformed.data = -data.ts^optimal.lambda 
training.data = window(transformed.data, start = 1918, end = 2002)
test.data = window(transformed.data, start = 2003)
orig.test.data = window(data.ts, start=2003)

# Exponential and double exponential smoothing

ses.model <- HoltWinters(training.data, beta=FALSE,gamma=FALSE)
des.model <- HoltWinters(training.data,gamma=FALSE)

pred.ses <- predict(ses.model,n.ahead=22)
pred.des <- predict(des.model,n.ahead=22)

apse.ses <- mean((pred.ses - test.data)^2)
apse.des <- mean((pred.des - test.data)^2)

apse.ses
apse.des

model <- HoltWinters(transformed.data, beta=FALSE, gamma=FALSE)
fitted.values <- model$fitted[,1]
forecast.values <- predict(model, n.ahead=5)
combined.values <- ts(c(fitted.values, forecast.values),
                      start=1918,end=2033, frequency=1)
combined.values <- (-combined.values)^(1/optimal.lambda)

plot(data.ts,xlim=c(1918,2033))
lines(combined.values, lwd=2, col="red")

## Not a good h-step predictor, good for 1-step prediction since it is just 
## a constant predictor (the mean), based on historical data.

## Residual diagnostics

# Regression

library(glmnet)
degrees <- 2:10
poly.t <- poly(as.vector(time(transformed.data)),15)
training.idx <- 1:85
test.idx <- 86:107

alphas <- c(0,0.25,0.5,0.75,1)

optimal_fit <- list()

for (a in alphas) {
  apse_values <- c()
  for (i in seq_along(degrees)) {
    p <- degrees[i]
    poly.training.time <- poly.t[-test.idx, 1:p]
    poly.test.time <- poly.t[-training.idx, 1:p]
    CV <- cv.glmnet(as.matrix(poly.training.time), as.numeric(training.data),
                    alpha=a,standardize=TRUE,intercept=TRUE)
    lambda <- CV$lambda.min
    predictions <- predict(CV, newx=as.matrix(poly.test.time), s="lambda.min")
    predictions <- (-predictions)^(1/optimal.lambda)
    squared_errors <- (predictions - orig.test.data)^2
    APSE <- mean(squared_errors)
    apse_values <- c(apse_values, APSE)
  }
  
  par(mfrow=c(1,2))
  plot(degrees, apse_values, main=paste("APSE vs Degree, Alpha = ", a))
  
  deg <- degrees[which.min(apse_values)]
  plot(training.data,
       main="Fitted Elastic Net Regression Model on Training Data")
  
  poly.training.time <- poly.t[-test.idx, 1:deg]
  CV <- cv.glmnet(as.matrix(poly.training.time), as.numeric(training.data), 
                  alpha=a,standardize=TRUE,intercept=TRUE)
  fitted_values <- predict(CV, newx=as.matrix(poly.training.time),s="lambda.min")
  lines(as.numeric(time(training.data)), fitted_values, col="red", lwd=2)
  optimal_fit[[length(optimal_fit) + 1]] <- list(model = CV, degree = deg)
}

apse_values <- c()
for (i in seq_along(optimal_fit)) {
  fit <- optimal_fit[[i]]$model
  deg <- optimal_fit[[i]]$degree
  predictions <- predict(fit, newx=as.matrix(poly.t[-training.idx, 1:deg]), s="lambda.min")
  predictions <- (-predictions)^(1/optimal.lambda)
  squared_error <- (predictions - orig.test.data)^2
  APSE <- mean(squared_error)
  apse_values <- c(apse_values, APSE)
  
  print(paste0("Alpha = ", alphas[i], ", Degree = ", deg, ", APSE = ", APSE))
}

opt_idx <- which.min(apse_values)
optimal.regression.model <- optimal_fit[[opt_idx]]$model
optimal.degree <- optimal_fit[[opt_idx]]$degree

print(paste0("The best model is Alpha = ", alphas[opt_idx],
             ", Degree = ", optimal.degree))

forecast.time <- 2025:2029
poly.forecast.time <- predict(poly.t, newdata=as.numeric(forecast.time))
fitted.values <- predict(optimal.regression.model, newx=as.matrix(poly.t[,1:optimal.degree]),s="lambda.min")
forecast.values <- predict(optimal.regression.model, newx=as.matrix(poly.forecast.time[,1:optimal.degree]),s="lambda.min")
combined.values <- ts(c(fitted.values, forecast.values), start=1918,end=2029, frequency=1)
combined.values <- (-combined.values)^(1/optimal.lambda)

plot(data.ts,xlim=c(1918,2029))
lines(combined.values, lwd=2, col="red")

## Some residual diagnostics on this fit.

residuals <- (-fitted.values)^(1/optimal.lambda) - data.ts

plot(residuals)
shapiro.test(residuals)
plot(fitted.values,residuals)
g <- cut(1:107, breaks=6, labels=1:6)
fligner.test(residuals, g)
acf(residuals)
randtests::difference.sign.test(residuals)
randtests::runs.test(residuals,plot=TRUE)

## Can use these residuals for ARIMA/SARIMA

## Robust Ridge Regression

lambda <- 10^(-4)

robust_ridge_loss <- function(beta, X, y) {
  residuals <- y - X %*% beta
  lad_loss <- mean(abs(residuals))
  penalty <- lambda * sum(beta[-1]^2)
  return (lad_loss + penalty)
}

apse_values <- c()

for (p in degrees) {
  poly.training.time <- poly.t[-test.idx, 1:p]
  poly.test.time <- poly.t[-training.idx, 1:p]
  
  initial_beta <- rep(1, ncol(poly.training.time) + 1)
  
  fit <- optim(par = initial_beta,
               fn=robust_ridge_loss,
               X=cbind(1, poly.training.time),
               y=training.data)
  
  beta <- fit$par
  predictions <- cbind(1, poly.test.time) %*% beta
  spredictions <- (-predictions)^(1/optimal.lambda)
  squared_errors <- (predictions - orig.test.data)^2
  APSE <- mean(squared_errors)
  apse_values <- c(apse_values, APSE)
}

plot(degrees, apse_values)
optimal_degree <- degrees[which.min(apse_values)]
print(paste0("The optimal degree is ", optimal_degree, " with APSE: ", min(apse_values)))

initial_beta <- rep(1, ncol(poly.t[,1:optimal_degree]) + 1)

fit <- optim(par = initial_beta,
             fn = robust_ridge_loss,
             X = cbind(1, poly.t[, 1:optimal_degree]),
             y = transformed.data)
beta <- fit$par

forecast.time <- 2025:2029
poly.forecast.time <- predict(poly.t, newdata=as.numeric(forecast.time))
fitted.values <- cbind(1, poly.t[,1:optimal_degree]) %*% beta
forecast.values <- cbind(1, poly.forecast.time[,1:optimal_degree]) %*% beta
combined.values <- ts(c(fitted.values, forecast.values), start=1918,end=2029, frequency=1)
combined.values <- (-combined.values)^(1/optimal.lambda)

plot(data.ts)
lines(combined.values, lwd=2, col="red")
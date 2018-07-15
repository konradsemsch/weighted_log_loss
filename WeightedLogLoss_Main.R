
### Setup

library(tidyverse)

data <- iris %>% 
  mutate(
    y  = as.numeric(ifelse(Species == "virginica", 1, 0)),
    cw = as.numeric(ifelse(Species == "virginica", 20, 1))
  )

table(data$y)

### ============ Implementing loss functions  ============ ###

# Logistic log loss
loss_glm <- function(param, df) {
  
  ### For tesing ###
  # param <- c(0, 0, 0)
  # df    <- data
  ###
  
  xb <- cbind(1 ,df$Sepal.Length, df$Petal.Length) %*% param
  p <- 1 / (1 + exp(-xb))
  sum(-(df$y * log(p) + (1 - df$y) * log(1 - p)))
}

######

# Logistic weighted log loss (cost refers to error on the virginica species)
loss_glm_cw <- function(param, df, weight = 10) {
  
  ### For tesing ###
  # param  <- c(0, 0, 0)
  # df     <- data
  # weight <- 10
  ###
  
  xb <- cbind(1 ,df$Sepal.Length, df$Petal.Length) %*% param
  p <- 1 / (1 + exp(-xb))
  sum(-((1 - (1 / weight)) * df$y * log(p) + (1 / weight) * (1 - df$y) * log(1 - p)))
}

######

# Elastic-net log loss
loss_en <- function(param, df, alpha = 0.5, lambda = 0.001) {
  
  ### For tesing ###
  # param  <- c(0, 0)
  # df     <- data
  # alpha  <- 0.5 
  # lambda <- 0.001
  ###
  
  xb <- cbind(df$Sepal.Length, df$Petal.Length) %*% param
  p <- 1 / (1 + exp(-xb))
  sum(-(df$y * log(p) + (1 - df$y) * log(1 - p)) + lambda * ((1 - alpha) / 2 * sum(param^2) + alpha * sum(abs(param))))
}

######
  
# Elastic-net weighted log loss (cost refers to error on the virginica species)
loss_en_cw <- function(param, df, alpha = 0.5, lambda = 0.001, weight = 20) {
  
  ### For tesing ###
  # param  <- c(0, 0)
  # df     <- data
  # alpha  <- 0.5
  # lambda <- 0.001
  # weight <- 20
  ###
  
  n_org <- nrow(df)
  n_cw  <- sum(ifelse(df$y == 1, weight, 1))
  
  cw_factor <- n_org / n_cw # calculate the normalization factor
  
  weights_norm <- df$cw * cw_factor # rescaling the desired weights to make sure to sum of weights is equal to the original sample size
  weights_norm <- sort(unique(weights_norm))
  
  xb <- cbind(df$Sepal.Length, df$Petal.Length) %*% param
  p <- 1 / (1 + exp(-xb))
  sum(-(weights_norm[[2]] * df$y * log(p) + weights_norm[[1]] * (1 - df$y) * log(1 - p)) + lambda * ((1 - alpha) / 2 * sum(param^2) + alpha * sum(abs(param))))
}

### ============ Testing against standard R implementations ============ ###

# Logistic log loss
optim(par = c(0, 0, 0), fn = loss_glm, df = data)$par
glm(y ~ Sepal.Length + Petal.Length, data, family = "binomial")

# Logistic weighted log loss
optim(par = c(0, 0, 0), fn = loss_glm_cw, df = data)$par
glm(y ~ Sepal.Length + Petal.Length, data, family = "binomial", weights = cw)

y <- data$y
x <- as.matrix(data[, c("Sepal.Length", "Petal.Length")])

# Elastic-net log loss
optim(par = c(0, 0), fn = loss_en, df = data)$par
glmnet::glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.001, standardize = FALSE, intercept = FALSE)$beta

# Elastic-net weighted log loss
optim(par = c(0, 0), fn = loss_en_cw, df = data)$par
glmnet::glmnet(x, y, family = "binomial", alpha = 0.5, lambda = 0.001, standardize = FALSE, intercept = FALSE, weights = data$cw)$beta
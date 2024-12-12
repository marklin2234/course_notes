## STAT 340 (Winter 2024)
## Tutorial 01
## Erik Hintz 

## Question 1: MWEs #######################################################

## A minimal working example (MWE) for a problem is an example that
##  - working (someone else can reproduce the problem)
##  - minimal (as simple as possible)
## They are used to show a problem you may have to somebody else. For instance, 
## you will find MWEs on stackexchange/... where people seek help with their issues,
## or when submitting bug reports to developpers.

## Somebody wants to compute A x (vector product), should be using A %*% x 
## INSTEAD of A * x 

### The following *is* a MWE:

A <- matrix(c(1, 2, 3), ncol = 3, nrow = 3) # note the recycling
x <- rep(1, 3) 
res <- A * x # here's the error
stopifnot(res == c(3, 6, 9)) # we know the true result of A*x, namely (3, 6, 9)

## This is a MWE because 
## - it is working (anybody can reproduce the errror)
## - it is minimal (it cannot be shorter or simpler)

### The following is *not* a MWE:

n <- 100
m <- 1000
A <- matrix(NA, ncol = m, nrow = n)
for(i in 1:n){
   for(j in 1:m){
      A[i, j] <- (-1)^i * i/j 
   }
}
x <- rep(1, m)
for(i in 3:m){
   x[i] <- sum(x[(i-2):(i-1)])
}
res <- A*x
dim(res) # 100, 1000 wheras it should be 100, 1 had one used ' %*% ' 

## This is not a MWE because it is not minimal: You don't need a complicated
## 100 x 1000 matrix 'A' that is filled with a for-loop to generate an example
## for matrix and you also don't need to use a complicated 'x' (here: Fibonacci
## sequence) as a vector to construct an example for a matrix - vector multi-
## plication. If possible t is better to take a small example where you know the 
## true answer, as above 

### The following is also *not* a MWE:

A <- load("mydataset")
x <- 1:100
A * x

## This is not a MWE because it's neither minimal, nor working: If someone
## does not have "mydataset", they cannot reproduce the error at all. 
## It is advisable to generate the example within R, if possible, for instance 
## using dummy values or even random numbers (=> later), rather than relying on 
## external data-sets. 

## Note: Plenty of hints and guidelines for MWEs can be found online!



## Question 2: Converting integers to binary reps ##############################

## See also Question 1.5 in the exercises file 

#' Compute k-bit binary representation from integer representation 
#'
#' @param n integer to be converted to binary
#' @param k number of bits 
#' @return vector of length k (x_{k-1},..,x_0) including binary representation
#'         of 'n' (or ERROR is n cannot be represented with k bits)
int_to_bit <- function(n, k){
   stopifnot(n >= 0, k > 0) # check
   x <- integer(k) # by default filled with '0' 
   for(i in 1:k){
      if(n == 0) return(x) 
      x[k-i+1] <- n %% 2 # remainder when dividing by 2 
      n <- floor(n/2) # divide n by 2, ignore remainder
   }
   if(n > 0) stop("n cannot be represented with k bits") 
   x # return 
}

## Test cases
stopifnot(all.equal(int_to_bit(1, k = 2), c(0, 1)),
          all.equal(int_to_bit(25, k = 7), c(0, 0, 1, 1, 0, 0, 1)))
## Check if error is thrown: n = 32 = 2^6 is not representable with 5 bits
errorTrue <- tryCatch(int_to_bit(32, k = 5), error = function(e) TRUE) 
stopifnot(errorTrue) # errorTrue is TRUE => 'int_to_bit()' indeed threw an error 



## Question 3: A simple example of a simulation problem ########################

## True probability
(p <- pnorm( (2.98-3)/sqrt(1.7) ) + 1 - pnorm( (3.02-3)/sqrt(1.7)))

## Version 1:

estimate_p_1 <- function(n){
   stopifnot(n > 0) # check
   counter <- 0 
   for(i in 1:n){
      L <- rnorm(1, mean = 1, sd = 1) + rnorm(1, mean = 1, sd = sqrt(0.5)) +
         rnorm(1, mean = 1, sd = sqrt(0.2))
      if(L < 2.98 || L > 3.02) counter <- counter + 1
   }
   counter / n
} 

set.seed(12) # reproducibility
estimate_p_1(5e3)


## PRGRAMMING HINT: Avoid hard coded values as MUCH AS YOU CAN!
## Imagine the code was a lot more complicated and involved a lot of parameters,
## and you'd like to perform a sensitivity analysis. Rather than going through the
## code each time and change the numerical value (=> time consuming, error prone)
## just use arguments to the function (here: the standard deviations of L1, L2, L3;
## and/or the means of L1, L2, L3). 
##
## PRGRAMMING HINT: We can also set default values for any function variable, and  
## I would recommended to do so for the non-important ones that don't change as 
## much. This makes the code when calling the function much shorter 
## (and again, less error prone!)

estimate_p_2 <- function(n, mu1 = 1, mu2 = 1, mu3 = 1, sig1 = 1, sig2 = sqrt(0.5), 
                         sig3 = sqrt(0.2)){
   stopifnot(n > 0) # check
   counter <- 0 
   for(i in 1:n){
      L <- rnorm(1, mean = mu1, sd = sig1) + rnorm(1, mean = mu2, sd = sig2) +
         rnorm(1, mean = mu3, sd = sig3)
      if(L < 2.98 || L > 3.02) counter <- counter + 1
   }
   counter / n
} 

set.seed(12) # reproducibility
estimate_p_2(5e3) # same as above, of course 

## PROGRAMMING HINT: Avoid for loops in R when you can, as they are typically 
## time consuming. 

estimate_p_3 <- function(n, mu1 = 1, mu2 = 1, mu3 = 1, sig1 = 1, sig2 = sqrt(0.5), 
                         sig3 = sqrt(0.2)){
   stopifnot(n > 0) # check
   L <- rnorm(n, mean = mu1, sd = sig1) + rnorm(n, mean = mu2, sd = sig2) +
       rnorm(n, mean = mu3, sd = sig3)
   mean(  (L < 2.98 | L > 3.02)  )
   ## Aside: 
   ## *) 'L < 2.98' and 'L > 3.02' are each n-vectors with element i indicating 
   ## whether 'L[i] < 2.98' (for the former) and 'L[i] > 3.02' (for the latter).
   ## *) The operator "|" applied to vectors works componentwise, i.e., 
   ## '(L < 2.98 | L > 3.02)' is again a n-vector indicating with element i 
   ## indicating whether 'L[i] < 2.98 or L[i] > 3.02'. 
   ## CAREFUL: The operator "||" (note: 2 '|'s )applies the OR operator on all 
   ## the elements to its left and right jointly, and just returns (one) 
   ## TRUE or FALSE. Similar for '&' and '&&'
   ## *) The function 'mean' interprets the n-vector (L < 2.98 | L > 3.02) of
   ## TRUEs and FALSEs as 1s and 0s, takes the sum and divides by it's length;
   ## i.o.w., it counts how often  'L[i] < 2.98 or L[i] > 3.02' is true, 
   ## and converts it into the relative frequency (between 0 and 1). 
} 

set.seed(12)
estimate_p_3(5e3) 
## Aside: This result is not exactly the same as above. 
## Why? It has to do with the order in which we sampled the observations. 
## In 'estimate_p_1()', 'estimate_p_2()' we sampled L1, then L2, then L3; then 
## L1 again, and so on. In 'estimate_p_3()', we first sampled L1, L1,..., L1,
## then all the L2's, then all the L3's; so we used the internal random numbers
## in a different order. Mathematically, it does not matter - the "distribution"
## of the estimators over many simulations will be the same. 

## ASIDE: Let's do some timing experiment: 
## => Is the for loop really that much slower?
## We use 'system.time()' and call the functions with a larger sample size 
## to reduce the noise in the numbers. 
set.seed(12)
system.time(estimate_p_2(1e6)) 
set.seed(12)
system.time(estimate_p_3(1e6))
## => *significantly* faster


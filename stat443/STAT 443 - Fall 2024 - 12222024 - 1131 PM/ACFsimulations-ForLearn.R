################################

#### Theoretical ACF and PACF ####

################################

#MA(1) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = numeric(0), ma = c(0.7), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1) # note that h starts at 0 for acf but at 1 for pacf
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = numeric(0), ma = c(0.7), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#MA(2) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = numeric(0), ma = c(0.7, -1), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = numeric(0), ma = c(0.7, -1), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#MA(3) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = numeric(0), ma = c(1, 1, 1), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = numeric(0), ma = c(1, 1, 1), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#AR(1) (theoretical)
par(mar=c())
par(mfrow=c(2,1))

ARMA.acf = ARMAacf(ar = c(0.7), ma = numeric(0), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = c(0.7), ma = numeric(0), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#AR(2) (theoretical) - different parameters from previous case
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = c(0.7, 0.1), ma = numeric(0), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = c(0.7, 0.1), ma = numeric(0), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#AR(4) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = c(0.7, 0.1, -0.2, 0.1), ma = numeric(0), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = c(0.7, 0.1, -0.2, 0.1), ma = numeric(0), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#ARMA(1,1) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = c(0.5), ma = c(1), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = c(0.5), ma = c(1), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)


#ARMA(2,1) (theoretical)
par(mfrow=c(2,1))
ARMA.acf = ARMAacf(ar = c(0.3,0.65), ma = c(-0.5), lag.max = 15, pacf = F)
h=0:(length(ARMA.acf)-1)
plot(h , ARMA.acf , main="ACF", ylab=expression(rho(h))  , col=adjustcolor("black" , 0.7) , pch=16)

ARMA.pacf = ARMAacf(ar = c(0.3,0.65), ma = c(-0.5), lag.max = 15, pacf = T)
plot(ARMA.pacf , main="PACF", ylab=expression(alpha(h)) , xlab="h"  , col=adjustcolor("black" , 0.7) , pch=16)

###################################################################
#### Sample estimates for ACF and PACF based on simulated data ####
###################################################################
#MA(1) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ma = c(0.7)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#MA(2) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ma = c(0.7,-1)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#MA(3) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ma = c(1,1,1)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#AR(1) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ar = c(0.7)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#AR(2) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ar = c(-0.7, 0.1)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#AR(4) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ar = c(0.7, 0.1, -0.2, 0.1)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#ARMA(1,1) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ar=c(0.5) , ma = c(1)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")

#ARMA(2,1) (estimates based on simulated data)
par(mfcol=c(3,1))
data.sim <- arima.sim(n = 200, list(ar=c(0.3,0.65) , ma = c(-0.5)), sd = sqrt(1))
plot(data.sim, main="Simulated Data")
acf(data.sim)
acf(data.sim, type="partial")



##############################
### Ma(2) basic simulation ###
##############################

theta1 <- 0.75 #define parameters
theta2 <- 0.3
n <- 500 #sample size
z <- rnorm(n) #Innovations
x <- seq(0, length=n)
x[1] <- z[1] #initial values
x[2] <- z[2]
for(i in 3:n) #generating output
{
  x[i] <- z[i] + theta1*z[i-1]+ theta2*z[i-2]
}

#plotting
par(mfcol=c(3,1))
plot(x, type='l', main="simulated MA(2) process")
acf(x, main="ACF of simulated MA(2) process")
pacf(x, main="PACF of simulated MA(2) process")


##########################
### AR(1) basic simulation ###
##########################
phi <- 2 #define parameters
n <- 100 #sample size
z <- rnorm(n) #Innovations
x <- seq(0, length=n)
x[1] <- z[1] #initial values
for(i in 2:n)
{
  x[i] <- phi*x[i-1]+ z[i]
}

#plotting
par(mfcol=c(3,1))
plot(x, type='l', main="simulated AR process")
acf(x, main="ACF of simulated AR process")
pacf(x, main="PACF of simulated AR process")



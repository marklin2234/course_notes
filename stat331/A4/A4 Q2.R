directory <- "~/Documents/waterloo/stat331/A4"
load(file.path(directory, "UScrime05.Rda"))

plot(UScrime05$violent, UScrime05$murder, main="Murder vs Violent", ylab="Murder", xlab="Violent")

fit1 <- lm(murder ~ violent, data=UScrime05)
summary(fit1)
abline(fit1)

qqtest(fit1)


std.resid <- rstandard(fit1)
sqrt.abs.std.resid <- sqrt(abs(std.resid))
plot(fit1$fitted.values, sqrt.abs.std.resid, 
     xlab = "Fitted Values", ylab = "standardized residuals",
     main = "Plot of standardized residuals vs Fitted Values")

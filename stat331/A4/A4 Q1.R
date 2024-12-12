# STAT 331 A4
directory <- "~/Documents/waterloo/stat331/A4"
load(file.path(directory, "loon.data/data/lizards.rda"))
lizards$sex <- factor(lizards$sex)
lizards$species <- factor(lizards$species)

plot(lizards$perchDiameter, lizards$perchHeight, main="Lizard Perch Height vs Perch Diameter", xlab="Perch Diameter", ylab="Perch Height")
fit1 <- lm(lizards$perchHeight ~ lizards$perchDiameter, data=lizards)
abline(fit1)

summary(fit1)

fit2 <- lm(lizards$perchHeight ~ lizards$perchDiameter + lizards$sex + lizards$perchDiameter:lizards$sex, data=lizards)
anova(fit1, fit2)

plot(lizards$perchDiameter, lizards$perchHeight, col=ifelse(lizards$sex=="male","steelblue","firebrick"), main="Lizard Perch Height vs Perch Diameter", xlab="Perch Diameter", ylab="Perch Height")
coefs_hat <- coefficients(fit2)
abline(coefs_hat[1], coefs_hat[2], lty=2)
abline(coefs_hat[1] + coefs_hat[3] , coefs_hat[2] + coefs_hat[4], lty=1)

fit3 <- lm(lizards$perchHeight ~ lizards$perchDiameter + lizards$species + lizards$perchDiameter:lizards$species, data=lizards)
anova(fit1, fit3)

species_colors <- c("firebrick", "steelblue", "orange", "seagreen")
plot(lizards$perchDiameter, lizards$perchHeight, col = species_colors[lizards$species], xlab = "Perch Diameter", ylab = "Perch Height",
     main = "Perch Height vs. Perch Diameter by Species")
coefs_hat <- coefficients(fit3)
abline(coefs_hat[1], coefs_hat[2], lty=1)
abline(coefs_hat[1] + coefs_hat[3], coefs_hat[2] + coefs_hat[6], lty=2)
abline(coefs_hat[1] + coefs_hat[4], coefs_hat[2] + coefs_hat[7], lty=3)
abline(coefs_hat[1] + coefs_hat[5], coefs_hat[2] + coefs_hat[8], lty=4)
legend("topleft", legend = unique(lizards$species), 
       col = species_colors[unique(lizards$species)], pch = 1)

fit4 <- lm(lizards$perchHeight ~ lizards$perchDiameter + lizards$species, data = lizards)
anova(fit4, fit3)

counts <- with(lizards, table(sex, species))
diameterAves <- with(lizards,
                     aggregate(perchDiameter ~ sex + species,
                               FUN = mean)
)
heightAves <- with(lizards,
                   aggregate(perchHeight ~ sex + species,
                             FUN = mean)
)
perchAves <- data.frame(
  perchHeight = heightAves$perchHeight,
  perchDiameter = diameterAves$perchDiameter,
  sex = heightAves$sex,
  species = heightAves$species,
  count = as.vector(counts)
)

# The weighted least squares model is
fit_wls <- lm(perchHeight ~ perchDiameter*sex,
              data = perchAves,
              weights = as.vector(counts))

colors <- ifelse(perchAves$sex == "male", "steelblue", "firebrick")
pch <- as.character(perchAves$species)
plot(perchHeight ~ perchDiameter, data = perchAves, col = colors, pch = pch, main="Perch Height vs Perch Diameter", xlab="Perch Diameter", ylab="Perch Height")
abline(lm(perchHeight ~ perchDiameter, data = perchAves, subset = sex == "female"), col = "firebrick")
abline(lm(perchHeight ~ perchDiameter, data = perchAves, subset = sex == "male"), col = "steelblue")

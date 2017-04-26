########################################
#
#
# INTRODUCCIÓN A LAS REDES NEURONALES
#
#    Manel Velasco
#
#       Abril, 2017
#
########################################

########################################
#
# LIBRERIAS
#
library(ggplot2)
library(scatterplot3d)
library(neuralnet)
library(NeuralNetTools)
library(nnet)
library(caret)
library(zoo)
library(quantmod) 
#library(forecast)



########################################
#
# Regresión Lineal
#
# 1 variable

a = 1
b = 1
x <- runif(1000)
y <- b + a*x + rnorm(1000,0,0.15)

ggplot() + 
  geom_point(aes(x=x, y=y, colour = I("blue"))) + 
  theme_bw()


modelo.lin <- lm(y~x)
summary(modelo.lin)

ggplot() + 
  geom_point(aes(x=x, y=y, colour = I("blue"))) + 
  geom_line(aes(x= x, y=predict(modelo.lin)), color = I("red"), size=1) +
  theme_bw() + ggtitle("Modelo Lineal con Datos Entrenamiento")


x.test <- data.frame(x = seq(0,1,by=0.01))
y.test <- b + a*x.test$x + rnorm(101,0,0.15)
y.lin.pred <- predict(lm(y ~ x), x.test, se.fit = TRUE)
error.rmse <- sqrt(mean((y.lin.pred$fit - y.test)^2)) 
print(error.rmse)

ggplot() + 
  geom_point(aes(x=x.test, y=y.test, colour = I("blue"))) + 
  geom_line(aes(x= x.test, y = y.lin.pred$fit), color = I("red"), size=1) +
  theme_bw() + ggtitle(paste("Modelo Lineal con Datos Test -", "Error Cuadrático Medio: ",error.rmse  ))


#########################################################
#  RESIDUOS


residuos <- resid(modelo.lin)
ggplot() + 
  geom_point(aes(x=x, y=residuos, colour = I("blue"))) + 
  theme_bw()

#########################################################
#  HISTOGRAMA
hist(residuos, col = 'pink', border = 'red', main = "Histograma de los residuos", prob=TRUE)
lines(density(residuos))


#########################################################
#  PRIMERA RED NEURONAL

datos.train <- data.frame(x,y)
modelo.rn <-neuralnet(y~x,data=datos.train, hidden=1, stepmax=2e05, threshold=0.02, lifesign="full")
plot(modelo.rn)
plotnet(modelo.rn)

y.pred <- compute(modelo.rn, x)$net.result
error.rmse <- sqrt(mean((y.pred - y)^2)) 
print(error.rmse)

ggplot() + 
  geom_point(aes(x=x, y=y, colour = I("blue"))) + 
  geom_line(aes(x= x, y = y.pred), color = I("red"), size=1) +
  theme_bw() + ggtitle(paste("RED NEURONAL con Datos Entrenamiento -", "Error Cuadrático Medio: ",error.rmse  ))


datos.test <- data.frame(x.test, y.test)
y.pred <- compute(modelo.rn, x.test)$net.result
error.rmse <- sqrt(mean((y.pred - y.test)^2)) 
print(error.rmse)

ggplot() + 
  geom_point(aes(x=x.test, y=y.test, colour = I("blue"))) + 
  geom_line(aes(x= x.test, y = y.pred), color = I("red"), size=1) +
  theme_bw() + ggtitle(paste("RED NEURONAL con Datos Test -", "Error Cuadrático Medio: ",error.rmse  ))

###########################################################################
#
# 2 variables
#

x <- runif(100)
z <- runif(100)
y <- 2*x + 3*z + rnorm(100, mean=0, sd=0.15)

modelo.lin <- lm(y~x+z)
residuos <- resid(modelo.lin)
plot(x,residuos)
plot(z,residuos)
summary(modelo.lin)
hist(residuos, col = 'pink', border = 'red', main = "Histograma de los residuos", prob=TRUE)
lines(density(residuos))



grafico3d <-scatterplot3d(x,z,y, pch=16, 
                    highlight.3d=TRUE, type="h", main="Sistema Lineal")

grafico3d$plane3d(modelo.lin)


#########################################################################
#
# SISTEMA NO LINEAL !!!!!
#

# x <- runif(1000)
# z <- runif(1000)
# y<-sin(3*pi*x)+cos(3*pi*z) + rnorm(1000, mean=0, sd=0.15)

set.seed(1231239)
x <- runif(1000)
z <- runif(1000)
y<-sin(3*pi*x)+cos(3*pi*z) + rnorm(1000, mean=0, sd=0.25)
datosNL <- data.frame(y,x,z)


modelo.lin <- lm(y~x+z)

plot(y~x, col = 'blue')
abline(modelo.lin,col='red',lwd=3)

plot(y~z, col = 'blue')
abline(modelo.lin,col='red',lwd=3)

residuos <- resid(modelo.lin)
plot(x,residuos)
plot(z,residuos)
summary(modelo.lin)
hist(residuos, col = 'pink', border = 'red', main = "Histograma de los residuos", prob=TRUE)
lines(density(residuos))


grafico3d <-scatterplot3d(x,z,y, pch=16, 
                          highlight.3d=TRUE, type="h", main="Sistema NO Lineal")
grafico3d$plane3d(modelo.lin)



# Probamos con 4 neuronas

modelo.rn4 <-neuralnet(y~x+z,data=datosNL, hidden=4, stepmax=2e05, threshold=0.02, lifesign="full")
plotnet(modelo.rn4)
plot(modelo.rn4)


x.test <-seq(0,1,by=0.01)
z.test <-seq(from=0, to=1, by=0.01)
y.test <-sin(3*pi*x.test)+cos(3*pi*z.test) + rnorm(length(x.test), mean=0, sd=0.25)
datosNL.Test = data.frame(y,x,z)

y.pred<-compute(modelo.rn4, 
                covariate=matrix(c(x.test, rep(0.5, length(x.test))), 
                                 nrow=length(x.test), ncol=2))$net.result
x.rmse <- sqrt(mean((y.pred - y.test)^2)) 
print(x.rmse)
plot(y~x, data=datosNL, 
     main=paste("RED NEURONAL 4 Neuronas", "Error Cuadrático Medio Test X: ", x.rmse  ) )
lines(y.pred~x.test, data=datosNL.Test, type="l", col="red", lwd=2)

y.predZ<-compute(modelo.rn4, covariate=matrix(c(z.test, rep(0, length(z.test)), z.test), 
                                              nrow=length(z.test), ncol=2))$net.result
z.rmse <- sqrt(mean((y.predZ - y.test)^2)) 
print(z.rmse)
plot(y~z, data=datosNL,
     main=paste("RED NEURONAL 4 Neuronas", "Error Cuadrático Medio Test Z: ", z.rmse  ) )
lines(y.predZ~z.test, type="l", col="red", lwd=2)


# Probamos con 8 neuronas

modelo.rn8 <-neuralnet(y~x+z,data=datosNL, hidden=8, stepmax=2e05, threshold=0.02, lifesign="full")
plotnet(modelo.rn8)
plot(modelo.rn8)


y.pred<-compute(modelo.rn8, 
                covariate=matrix(c(x.test, rep(0.5, length(x.test))), 
                                 nrow=length(x.test), ncol=2))$net.result

x.rmse <- sqrt(mean((y.pred - y.test)^2)) 
print(x.rmse)
plot(y~x, data=datosNL, 
     main=paste("RED NEURONAL 8 Neuronas", "Error Cuadrático Medio Test X: ", x.rmse  ) )
lines(y.pred~x.test, data=datosNL.Test, type="l", col="red", lwd=2)



y.predZ<-compute(modelo.rn8, covariate=matrix(c(z.test, rep(.5, length(z.test)), z.test), 
                                              nrow=length(z.test), ncol=2))$net.result
z.rmse <- sqrt(mean((y.predZ - y.test)^2)) 
print(z.rmse)
plot(y~z, data=datosNL,
     main=paste("RED NEURONAL 8 Neuronas", "Error Cuadrático Medio Test Z: ", z.rmse  ) )
lines(y.predZ~z.test, type="l", col="red", lwd=2)



########################################
#
# Regresión Logística
#
data(iris)
sel <-which(1:length(iris[,1])%%5==0) 
iris.train <- iris[-sel,]
iris.test <- iris[sel,]

colSetosa = data.frame(esSetosa=(iris.train$Species == 'setosa'))
colVersicolor = data.frame(esVersicolor=(iris.train$Species == 'versicolor'))
colVirginica = data.frame(esVirginica=(iris.train$Species == 'virginica'))


train.df <- cbind(iris.train, colSetosa,colVersicolor,colVirginica)

modelo.log.setosa <- glm(esSetosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                   data=train.df, family='binomial')
modelo.log.versicolor <- glm(esVersicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                         data=train.df, family='binomial')
modelo.log.virginica <- glm(esVirginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
                         data=train.df, family='binomial')


clasifica.setosa <- predict(modelo.log.setosa, newdata=iris.test, type='response')
clasifica.versicolor <- predict(modelo.log.versicolor, newdata=iris.test, type='response')
clasifica.virginica <- predict(modelo.log.virginica, newdata=iris.test, type='response')

print(round(clasifica.setosa, 0))
print(round(clasifica.versicolor, 0))
print(round(clasifica.virginica, 0))

########################################
#
# Perceptrón
#

# Puertas Lógicas

x = c(rep(0,4),rep(1,4))
y = c(rep(c(0,0,1,1),2))
z = c(rep(c(0,1),4))
nor = c(1,rep(0,7))
nand = c(rep(1,7),0)
or = c(0,rep(1,7))
and = c(rep(0,7),1)

datos.train = data.frame(x,y,z,nand,and,nor,or)
datos.test = data.frame(x,y,z)

print(datos.train)
print(datos.test)

nand.rn <-neuralnet(nand~x+y+z, data=datos.train, 
                    hidden=1, stepmax=2e05, threshold=0.02, lifesign="full")
plot(nand.rn)
and.rn <-neuralnet(and~x+y+z, data=datos.train, 
                   hidden=1, stepmax=2e05, threshold=0.02, lifesign="full")
plot(and.rn)

nor.rn <-neuralnet(nor~x+y+z, data=datos.train, 
                   hidden=1, stepmax=2e05, threshold=0.02, lifesign="full")
plot(nor.rn)

or.rn <-neuralnet(or~x+y+z, data=datos.train, 
                  hidden=1, stepmax=2e05, threshold=0.02, lifesign="full")
plot(or.rn)

nand.result <- ifelse(compute(nand.rn, datos.test)$net.result>0.5, 1, 0)
and.result <- ifelse(compute(and.rn, datos.test)$net.result>0.5, 1, 0)
nor.result <- ifelse(compute(nor.rn, datos.test)$net.result>0.5, 1, 0)
or.result <- ifelse(compute(or.rn, datos.test)$net.result>0.5, 1, 0)

resultados <- data.frame(x,y,z,nand.result, and.result, nor.result, or.result)

print(resultados)




##########################################################
#
# Series temporales
#

set.seed(1231239)
horizonte = 6
fin.train = 2*pi
fin.test  = 4*pi

t.train = seq(0, fin.train,by=0.01)
t.test  = seq(fin.train, fin.test,by=0.01)
t = c(t.train,t.test)
y.train = 5*sin(t.train) + 2*sin(2*pi*t.train) +  rnorm(length(t.train), sd=0.25)
y.test  = 5*sin(t.test) + 2*sin(2*pi*t.test) +  rnorm(length(t.test), sd=0.25)
y = c(y.train, y.test)

plot(t,y, type = 'l')

stTrainDF <- data.frame( y.train, x1=Lag(y.train, horizonte+1), x2=Lag(y.train,horizonte+2))
stTestDF  <- data.frame( y.test, x1=Lag(y.test,horizonte+1), x2=Lag(y.test, horizonte+2))

stTrainDF = stTrainDF[-c(1:(horizonte+2)),]
stTestDF = stTestDF[-c(1:(horizonte+2)),]

names(stTrainDF) <- c('y','x1','x2')
names(stTestDF) <- c('y','x1','x2')

modelo.st <-neuralnet(y ~ x1+x2, data=stTrainDF, 
                      hidden=8, rep = 1, 
                      stepmax=2e05, threshold=0.02, lifesign="full")

pst.train <- compute(modelo.st, stTrainDF[,-1])$net.result
pst.test <- compute(modelo.st, stTestDF[,-1])$net.result


plot(t,y,type="l",col = 'black')
lines(t.train[-c(1:(horizonte+2))],pst.train, col='blue')
lines(t.test[-c(1:(horizonte+2))],pst.test, col='red')
abline(v=fin.train)
legend(5, 70, c("y", "pred"), cex=1.5, fill=2:3)




##########################################################
#
# Reconocimiento de imagenes
#

# Muestra una imagen de 28x28 pixels
mostrarImagen <- function(arr784, col=gray(12:1/12), ...) {
  par(mfrow=c(1,1))
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# Muestra 20 imagenes de 28x28 pixels
mostrar20imagenes <- function(arr20, col=gray(12:1/12), ...) {
  par(mfrow=c(4,5))
  for (i in 1:20){
    image(matrix(arr20[i,], nrow=28)[,28:1], col=col, ...)
  }
}



# Cargar los ficheros MNIST

cargar_ficheros_imagenes <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

cargar_ficheros_valores <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

# Cargar todos los ficheros y sus etiquetas

train <- cargar_ficheros_imagenes('/home/manel/Proyectos/R/redesNeuro/train-images.idx3-ubyte')
test <- cargar_ficheros_imagenes('/home/manel/Proyectos/R/redesNeuro/t10k-images.idx3-ubyte')

train$y <- cargar_ficheros_valores('/home/manel/Proyectos/R/redesNeuro/train-labels.idx1-ubyte')
test$y <- cargar_ficheros_valores('/home/manel/Proyectos/R/redesNeuro/t10k-labels.idx1-ubyte')  

# MOstrar una imagen cualquiera
mostrarImagen(train$x[2,])

# Escogo solo los unos y los treses
sontres = train$y==3
tres = train$x[sontres,]
sonuno = train$y==1
uno = train$x[sonuno,]

# Los datos son una mezcla de unos y treses
datos.train = rbind(tres[1:10,],uno[1:10,])
datos.test = rbind(tres[11:20,],uno[11:20,])

# Muestro los datos de entrenamiento y test
mostrar20imagenes(datos.train)
mostrar20imagenes(datos.test)

# Añado las etiquetas
valor=c(rep(1,10),rep(0,10))
datos.train = data.frame(datos.train,valor)
nombreCol <- colnames(datos.train[1:784])
colnames(datos.test) <- nombreCol

# Entreno red Neuronal

reconoceTres <- train(datos.train[,1:784], datos.train[,785], method = "nnet", maxit = 100)

# Reconocimiento de una imagen
reconoceImagen <- function(miImg){
  
  img.df <- as.data.frame(t(miImg))
  colnames(img.df) <- nombreCol
  pred <- predict(reconoceTres, img.df)
  if (pred > 0.5) tit = "ES UN TRES"
  else tit = "ES UN UNO"
  image(matrix(miImg, nrow=28)[,28:1], col=gray(12:1/12), main=tit)
}

# Prediccion todas las imagenes Test
par(mfrow=c(5,4))
for (i in 1:nrow(datos.test)){
  reconoceImagen(datos.test[i,])  
}
par(mfrow=c(1,1))

reconoceImagen(tres[35,])
reconoceImagen(uno[78,])






















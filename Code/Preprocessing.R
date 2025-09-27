dataviz <- read.csv("GeneratedData//fish_fun_fitness.csv")
dataviz_1 <- read.csv("GeneratedData//bat_fun_fitness.csv")
dataviz_2 <- read.csv("GeneratedData//wolf_fun_fitness.csv")
dataviz_3 <- read.csv("GeneratedData//bee_fun_fitness.csv")


datawolf<-data.frame(matrix(nrow=0,ncol=3))
colnames(datawolf)<-c("Iteration","Best","Average")
listIt=list()
countIt=0
for(i in 1:nrow(dataviz_2)){
  if(dataviz_2$Iteration[i]==countIt){
    listIt=append(listIt,dataviz_2$Fitness[i])
  }
  else{
    av=mean(unlist(listIt))
    mi=min(unlist(listIt))
    datawolf[countIt,]=c(countIt,mi,av)
    listIt=list()
    countIt=countIt+1
    listIt=append(listIt,dataviz_2$Fitness[i])
  }
  
}
write.csv(datawolf,"GeneratedData\\datawolf.csv")
datawolf

databee<-data.frame(matrix(nrow=0,ncol=3))
colnames(databee)<-c("Iteration","Best","Average")
listIt=list()
countIt=0
for(i in 1:nrow(dataviz_3)){
  if(dataviz_3$Iteration[i]==countIt){
    listIt=append(listIt,dataviz_3$Fitness[i])
  }
  else{
    av=mean(unlist(listIt))
    mi=min(unlist(listIt))
    databee[countIt,]=c(countIt,mi,av)
    listIt=list()
    countIt=countIt+1
    listIt=append(listIt,dataviz_3$Fitness[i])
  }
  
}
write.csv(databee,"GeneratedData\\databee.csv")
databee

databat<-data.frame(matrix(nrow=0,ncol=3))
colnames(databat)<-c("Iteration","Best","Average")
listIt=list()
countIt=0
for(i in 1:nrow(dataviz_1)){
  if(dataviz_1$Iteration[i]==countIt){
    listIt=append(listIt,dataviz_1$Fitness[i])
  }
  else{
    av=mean(unlist(listIt))
    mi=min(unlist(listIt))
    databat[countIt,]=c(countIt,mi,av)
    listIt=list()
    countIt=countIt+1
    listIt=append(listIt,dataviz_1$Fitness[i])
  }
  
}
write.csv(databat,"GeneratedData\\databat.csv")
databat


datafish<-data.frame(matrix(nrow=0,ncol=3))
colnames(datafish)<-c("Iteration","Best","Average")
listIt=list()
countIt=0
for(i in 1:nrow(dataviz)){
  if(dataviz$Iteration[i]==countIt){
    listIt=append(listIt,dataviz$Fitness[i])
  }
  else{
    av=mean(unlist(listIt))
    mi=min(unlist(listIt))
    datafish[countIt,]=c(countIt,mi,av)
    listIt=list()
    countIt=countIt+1
    listIt=append(listIt,dataviz$Fitness[i])
  }
  
}
write.csv(datafish,"GeneratedData\\datafish.csv")
datafish

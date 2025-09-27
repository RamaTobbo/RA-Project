
library(ggplot2)
library(gganimate)
library(gifski)
datawolf<-data.frame(matrix(nrow=0,ncol=3))
write.csv(datawolf,"GeneratedData\\test.csv")
dataviz <- read.csv("GeneratedData//datafish.csv")
dataviz_1 <- read.csv("GeneratedData//databat.csv")
dataviz_2 <- read.csv("GeneratedData//datawolf.csv")
dataviz_3 <- read.csv("GeneratedData//databee.csv")


selected_columns <- c(dataviz$Best,dataviz_1$Best, dataviz_2$Best, dataviz_3$Best)

max_value <- max(selected_columns)
min_value <- min(selected_columns)

selected_columns1 <- c(dataviz$Average, dataviz_1$Average, dataviz_2$Average, dataviz_3$Average)

max_value1 <- max(selected_columns1)
min_value1 <- min(selected_columns1)


max_iter <- max(dataviz$Iteration)

p <-  
  ggplot(
    dataviz,
    aes(Iteration, Best)
  ) +
  geom_line(size = 2,color="brown1")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value,max_value),breaks=seq(min_value,max_value))

p1 <-  
  ggplot(
    dataviz_1,
    aes(Iteration, Best)
  ) +
  geom_line(size = 2,color="purple")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value,max_value),breaks=seq(min_value,max_value))

p2 <-  
  ggplot(
    dataviz_2,
    aes(Iteration, Best)
  ) +
  geom_line(size = 2,color="royalblue")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value,max_value),breaks=seq(min_value,max_value))

p3 <-  
  ggplot(
    dataviz_3,
    aes(Iteration, Best)
  ) +
  geom_line(size = 2,color="orange")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value,max_value),breaks=seq(min_value,max_value))

p4 <-  
  ggplot(
    dataviz,
    aes(Iteration, Average)
  ) +
  geom_line(size = 2,color="brown1")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value1,max_value1),breaks=seq(min_value1,max_value1))

p5 <-  
  ggplot(
    dataviz_1,
    aes(Iteration, Average)
  ) +
  geom_line(size = 2,color="purple")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value1,max_value1),breaks=seq(min_value1,max_value1))

p6 <-  
  ggplot(
    dataviz_2,
    aes(Iteration, Average)
  ) +
  geom_line(size = 2,color="royalblue")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value1,max_value1),breaks=seq(min_value1,max_value1))

p7 <-  
  ggplot(
    dataviz_3,
    aes(Iteration, Average)
  ) +
  geom_line(size = 2,color="orange")+
  scale_x_continuous(limits = c(0, max_iter))+
  scale_y_continuous(limits=c(min_value1,max_value1),breaks=seq(min_value1,max_value1))

p

#p
an <- p + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a <- animate(an, renderer = gifski_renderer())

#p1
an1 <- p1 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a1 <- animate(an1, renderer = gifski_renderer())

#p2
an2 <- p2 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a2 <- animate(an2, renderer = gifski_renderer())

#p3
an3 <- p3 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a3 <- animate(an3, renderer = gifski_renderer())

#p4
an4 <- p4 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a4 <- animate(an4, renderer = gifski_renderer())

#p5
an5 <- p5 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a5 <- animate(an5, renderer = gifski_renderer())

#p6
an6 <- p6 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a6 <- animate(an6, renderer = gifski_renderer())

#p7
an7 <- p7 + 
  geom_point(aes(group = seq_along(Iteration)),size=3) +
  transition_reveal(Iteration) 

a7 <- animate(an7, renderer = gifski_renderer())

#saving
anim_save("Fish_Fun_Best.gif", animation = a,path="GeneratedData")
anim_save("Bat_Fun_Best.gif", animation = a1,path="GeneratedData")
anim_save("Wolf_Fun_Best.gif", animation = a2,path="GeneratedData")
anim_save("Bee_Fun_Best.gif", animation = a3,path="GeneratedData")
anim_save("Fish_Fun_Average.gif", animation = a4,path="GeneratedData")
anim_save("Bat_Fun_Average.gif", animation = a5,path="GeneratedData")
anim_save("Wolf_Fun_Average.gif", animation = a6,path="GeneratedData")
anim_save("Bee_Fun_Average.gif", animation = a7,path="GeneratedData")


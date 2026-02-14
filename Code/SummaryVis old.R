library(ggplot2)
library(gganimate)
library(gifski)

dataviz <- read.csv("C://Users//admin//Desktop//RawaneJoe_FinalProject//MichalewiczData//fish_mic_fitness.csv")


dataviz1 <- na.omit(dataviz)
dataviz1$Iteration<-dataviz1$Iteration

p <-  
  ggplot(
  dataviz1,
  aes(Iteration, Best)
) +
  geom_line(size = 2,color="brown1")+
  scale_x_continuous(limits = c(0, 49))+
  scale_y_continuous(limits=c(-1.9,0.0),breaks=seq(-1.8,0.0))



p
an <- p + 
  geom_point(aes(group = seq_along(Iteration)),size=1.2) +
  transition_reveal(Iteration) 


a <- animate(an, renderer = gifski_renderer())

a

anim_save("Fish_Mic_Best.gif", animation = a)



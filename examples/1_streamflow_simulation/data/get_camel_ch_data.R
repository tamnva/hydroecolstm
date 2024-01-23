library(magick)
library(magrittr)

setwd("C:/Users/nguyenta/Documents")
list.files(path='C:/Users/nguyenta/Documents', pattern = '*.PNG', full.names = TRUE) %>% 
  image_read() %>% # reads each path file
  image_join() %>% # joins image
  image_animate(fps=1) %>% # animates, can opt for number of loops
  image_write("FileName.gif") # write to current dir


# Set workding directory
setwd("C:/Users/nguyenta/Documents/GitHub/hydroModLSTM/examples/example_1")

# ------------------------------------------------------------------------------------
# Read and filter time series data
# ------------------------------------------------------------------------------------
train_period <- c("2011-01-01", "2020-12-31")


fileList <- list.files("./camels_ch/timeseries/observation_based")
check <- matrix(data = rep(NA, length(fileList)*10), ncol = 10)
selectFileList <- c()
counter <- 0

for (i in 1:length(fileList)){
  data <- read.csv(file = paste0("./camels_ch/timeseries/observation_based/", fileList[i]),
                   header = TRUE, sep =",")
  
  istart <- which(data[,1]==train_period[1])
  iend <- which(data[,1]==train_period[2])
  
  if (length(which(is.na(data[istart:iend,3]))) < 0.7*(iend-istart)){
    name <- gsub(".csv", "", fileList[i])
    name <- strsplit(name,"_")[[1]]
    name <- name[length(name)]
    gauge_id <- as.integer(rep(name,nrow(data)))
    data <- cbind(gauge_id, data)
    counter = counter + 1
    if (counter == 1){
      data$gauge_id <- 
      timeSeries <- data
    } else {
      timeSeries <- rbind(timeSeries, data)
    }
  }
}

name <- c("object_id", "time", "discharge_vol_m3_s", "discharge_spec_mm_d",
          "waterlevel_m", "precipitation_mm_d", "temperature_min_degC" , 
          "temperature_mean_degC", "temperature_max_degC", "rel_sun_dur",
          "swe_mm")
colnames(timeSeries) = name
write.csv(timeSeries, file = "time_series.csv", row.names = FALSE, quote=FALSE, na = "")
name <- data.frame(name = unique(timeSeries$object_id))
write.csv(name, file = "name.csv", row.names = FALSE, quote=FALSE, na = "")


# ------------------------------------------------------------------------------------
# Read write static attributes
# ------------------------------------------------------------------------------------
name <- read.csv(file = "name.csv", header = TRUE)
fileList <- list.files("./camels_ch/static_attributes")

for (i in 1:length(fileList)){
  data <- read.csv(file = paste0("./camels_ch/static_attributes/", fileList[i]),
                   header = TRUE, sep =",", skip = 1)
  # check if id is the same
  if (i == 1){
    object_id = data$gauge_id
  }
  
  if (all(object_id - data$gauge_id == 0)){
    if (i == 1){
      static <- data[,-c(2,3)]
    } else {
      static <- cbind(static, data[,-c(1)])
    }
  }
  print(i)
}

temp <- c()
for (i in 1:nrow(name)){
  print(i)
  temp <- c(temp, which(name[i,] == static$gauge_id))
}

write.csv(static[temp,], file = "static_attributes.csv", row.names = FALSE, quote=FALSE, na = "")


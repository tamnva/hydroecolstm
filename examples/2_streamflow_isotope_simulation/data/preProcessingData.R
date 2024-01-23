# This function is used for getting the IsotopeData in the Fyw package

  dataURL_v1 <- paste0("https://www.envidat.ch/dataset/6a2fefc6-ce6d-4eb7-9ea9-c3",
                       "7b64959437/resource/8a06beab-3092-45d3-ad41-a59d79e6cb71/",
                       "download/alptal_isotopes_daily_2015-2019.txt")
  
  # Read data
  isotopeData_v1 <- read.csv(dataURL_v1, header = TRUE, sep = ",")
  isotopeData_v1$date <- as.Date(isotopeData_v1$date, format = "%Y-%m-%d")
  
  # Download data version 2
  dataURL_v2 <- paste0("https://www.envidat.ch/dataset/6a2fefc6-ce6d-4eb7-9ea9-",
                       "37b64959437/resource/7865517d-357b-4e08-8b6a-b405eca4e547",
                       "/download/alptal_isotopes_daily_2015-2019_v2.txt")
  # Read data
  isotopeData_v2 <- read.csv(dataURL_v2, header = TRUE, sep = ",")
  isotopeData_v2$date <- as.Date(isotopeData_v2$date, format = "%Y-%m-%d")
  isotopeData_v2$isotopes_data_quality[which(is.na(isotopeData_v2$isotopes_data_quality))] <- 2

  write.csv(isotopeData_v1, "C:/Users/nguyenta/Documents/GitHub/isotopeData_v1.csv", row.names = FALSE)
  write.csv(isotopeData_v2, "C:/Users/nguyenta/Documents/GitHub/isotopeData_v2.csv", row.names = FALSE)
  
  #-------------------------------------------------------------------------------
  # Extract data - Alp
  #-------------------------------------------------------------------------------
  # Streamflow Q, precipitation P, isotope in Q (isoQ) and P (isoP) in the Alp
  
  meterological <- read.csv("C:/Users/nguyenta/Documents/GitHub/time_series.csv", header = TRUE, sep = ",")
  meterological <- subset(meterological, object_id = "2609")
  meterological$time <- as.Date(meterological$time, format = "%Y-%m-%d")
  meterological = meterological[5480:7305,]
  
  QO <- subset(isotopeData_v1, catchment == "Alp" & source == "Streamwater")
  QO <- QO[, c("date","waterflux_measured", "delta_18O")]
  
  PO <- subset(isotopeData_v2, catchment == "Alp" & source == "Precipitation")
  PO <- PO[, c("date", "precipitation_interpolated", "delta_18O")]
  
  RHT <- subset(isotopeData_v2, catchment == "Erlenbach" & source == "Precipitation")
  RHT <- RHT[, c("date", "rel_humidty", "air_temperature")]
 
  
  date = seq(from = as.Date("2015/01/01"), to = as.Date("2019/12/31"), by = "days")
  streamflow = rep(NA, length(date))
  oxigen_isotope_streamflow = streamflow
  precipitation = meterological$precipitation_mm_d
  oxigen_isotope_precipitation = streamflow
  relative_humidity = streamflow
  temperature = meterological$temperature_mean_degC

  idx = which(date %in% QO$date)
  streamflow[idx] = QO$waterflux_measured
  oxigen_isotope_streamflow[idx] = QO$delta_18O
  
  idx = which(date %in% PO$date)
  precipitation[idx] = PO$precipitation_interpolated
  oxigen_isotope_precipitation[idx] = PO$delta_18O
  
  idx = which(date %in% RHT$date)
  relative_humidity[idx] = RHT$rel_humidty
  temperature[idx] = RHT$air_temperature
  
  results <- data.frame(object_id = rep("Alp",length(date)),
                        time = date,
                        streamflow = streamflow,
                        oxigen_isotope_streamflow = oxigen_isotope_streamflow,
                        precipitation = precipitation,
                        oxigen_isotope_precipitation = oxigen_isotope_precipitation,
                        relative_humidity = relative_humidity,
                        temperature = temperature)
  
  #Lieariterox
  x = 1:nrow(results)
  y = oxigen_isotope_precipitation
  idx <- which(is.na(results$oxigen_isotope_precipitation))
  xna <- x[-idx]
  yna <- y[-idx]
  out <- data.frame(approx(xna,yna, x, method = "linear"))
  plot(out$y, results$oxigen_isotope_precipitation, type="p")
  results$oxigen_isotope_precipitation = out$y
  
  results$time <- paste0(results$time," 00:00")
  
  alp <- results
  #write.csv(results[170:1610,], "C:/Users/nguyenta/Documents/GitHub/alp.csv", row.names = FALSE,quote = FALSE) 
  

  #-------------------------------------------------------------------------------
  # Extract data - Erlenbach
  #-------------------------------------------------------------------------------
  # Streamflow Q, precipitation P, isotope in Q (isoQ) and P (isoP) in the Alp
  
  
  QO <- subset(isotopeData_v1, catchment == "Erlenbach" & source == "Streamwater")
  QO <- QO[, c("date","waterflux_measured", "delta_18O")]
  
  PO <- subset(isotopeData_v2, catchment == "Erlenbach" & source == "Precipitation")
  PO <- PO[, c("date", "precipitation_interpolated", "delta_18O")]
  
  RHT <- subset(isotopeData_v2, catchment == "Erlenbach" & source == "Precipitation")
  RHT <- RHT[, c("date", "rel_humidty", "air_temperature")]
  
  
  date = seq(from = as.Date("2015/01/01"), to = as.Date("2019/12/31"), by = "days")
  streamflow = rep(NA, length(date))
  oxigen_isotope_streamflow = streamflow
  precipitation = meterological$precipitation_mm_d
  oxigen_isotope_precipitation = streamflow
  relative_humidity = streamflow
  temperature = meterological$temperature_mean_degC
  
  idx = which(date %in% QO$date)
  streamflow[idx] = QO$waterflux_measured
  oxigen_isotope_streamflow[idx] = QO$delta_18O
  
  idx = which(date %in% PO$date)
  precipitation[idx] = PO$precipitation_interpolated
  oxigen_isotope_precipitation[idx] = PO$delta_18O
  
  idx = which(date %in% RHT$date)
  relative_humidity[idx] = RHT$rel_humidty
  temperature[idx] = RHT$air_temperature
  
  results <- data.frame(object_id = rep("Erlenbach",length(date)),
                        time = date,
                        streamflow = streamflow,
                        oxigen_isotope_streamflow = oxigen_isotope_streamflow,
                        precipitation = precipitation,
                        oxigen_isotope_precipitation = oxigen_isotope_precipitation,
                        relative_humidity = relative_humidity,
                        temperature = temperature)
  
  #Lieariterox
  x = 1:nrow(results)
  y = oxigen_isotope_precipitation
  idx <- which(is.na(results$oxigen_isotope_precipitation))
  xna <- x[-idx]
  yna <- y[-idx]
  out <- data.frame(approx(xna,yna, x, method = "linear"))
  plot(out$y, results$oxigen_isotope_precipitation, type="p")
  results$oxigen_isotope_precipitation = out$y
  results$time <- paste0(results$time," 00:00")
  
  isotopeData <- rbind(results[170:1610,], alp[170:1610,])
  write.csv(isotopeData, "C:/Users/nguyenta/Documents/GitHub/isotope.csv", row.names = FALSE,quote = FALSE) 
  
  


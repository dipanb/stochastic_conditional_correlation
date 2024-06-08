library(rmgarch)
library(PerformanceAnalytics)
library(quantmod)
library(rugarch)
library(car)
library(FinTS)
library(ggplot2)
library(reshape2)
library(dplyr)
library(MSGARCH)


setwd("C://Users//dipan//OneDrive//Desktop//LBS Courses//Business Project")
prices = read.csv('price_data_stocks.csv')

dates = prices[,1]
prices = prices[,-1]

n_asset = ncol(prices)

returns = as.data.frame(apply(prices, 2, function(x) diff(x) / lag(x)[-1]))
# returns = returns[1:nrow(returns)-1,]
# dates = dates[-which(rowSums(is.na(returns)) == 1)]
# prices = prices[-which(rowSums(is.na(returns)) == 1),]
# returns = returns[-which(rowSums(is.na(returns)) == 1),]
dates = dates[-1]

#Pearson Correlations and rolling std
pearson_corr = data.frame(Date = dates[120:length(dates)])
for (i in 1:ncol(returns)) {
  for (j in 1:ncol(returns)) {
    pearson_corr[,paste(colnames(returns)[i],'&',colnames(returns)[j])] = rollapply(returns, width = 120, 
                                                                                    FUN = function(x) cor(x[, i], x[, j]), 
                                                                                    by.column = FALSE, align = "right")
  }
}

pearson_std = data.frame(Date = dates[120:length(dates)])
for (i in 1:ncol(returns)) {
  pearson_std[,paste(colnames(returns)[i])] = rollapply(returns, width = 120, 
                                                        FUN = function(x) sd(x[, i]), 
                                                        by.column = FALSE, align = "right")
}

melted_data <- melt(pearson_corr[,c('Date','SP500 & STOXX',
                                    'SP500 & HSI',
                                    'SP500 & BOVESPA',
                                    'SP500 & NIFTY50')], id.vars = "Date")
melted_data$Date = as.Date(melted_data$Date)
ggplot(melted_data, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  labs(title = "Pearson Correlation",
       x = "Date",
       y = "Value",
       color = "Variable") +
  theme_minimal()+
  theme(legend.position = "top")


#DCC GARCH correlations
garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                          variance.model = list(garchOrder = c(1,1), 
                                                model = "sGARCH"), 
                          distribution.model = "norm")
dcc.garch11.spec = dccspec(uspec = multispec(replicate(n_asset, garch11.spec) ), 
                           dccOrder = c(1,1), 
                           distribution = "mvnorm")
dcc.garch11.spec
dcc.fit = dccfit(dcc.garch11.spec, data = na.omit(returns))

ts.plot(rcor(dcc.fit)[1,2,])


a = aperm(rcor(dcc.fit), c(3,1,2))
a = matrix(a, ncol = n_asset*n_asset)
DCC_corr = data.frame(Date = dates[1:length(dates)])
DCC_corr = cbind(DCC_corr, as.data.frame(a, col.names = colnames(pearson_corr)[2:(n_asset*n_asset+1)]))
colnames(DCC_corr) = colnames(pearson_corr)

melted_data <- melt(DCC_corr[,c('Date','SP500 & STOXX',
                                'SP500 & HSI',
                                'SP500 & BOVESPA',
                                'SP500 & NIFTY50')], id.vars = "Date")
melted_data$Date = as.Date(melted_data$Date)
ggplot(melted_data, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  labs(title = "DCC Correlation",
       x = "Date",
       y = "Value",
       color = "Variable") +
  theme_minimal()+
  theme(legend.position = "top")


DCC_std = cbind(data.frame(Date = dates[1:length(dates)]), as.data.frame(sigma(dcc.fit)))
DCC_Std = data.frame(DCC_std, row.names = NULL)

#DCC Copula
uspec = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                   variance.model = list(garchOrder = c(1,1), 
                                         model = "gjrGARCH", 
                                         variance.targeting=FALSE), 
                   distribution.model = "norm")
spec1 = cgarchspec(uspec = multispec( replicate(n_asset, uspec) ), 
                   asymmetric = TRUE,  
                   distribution.model = list(copula = "mvt", 
                                             method = "Kendall", 
                                             time.varying = TRUE, 
                                             transformation = "parametric"))

fit1 = cgarchfit(spec1, data = na.omit(returns), 
                 cluster = NULL, solver.control=list(trace=1))
a = lapply(fit1@mfit$Rt, function(x) c(x))
a = as.data.frame(t(as.data.frame(do.call(cbind, a))))
cGARCH_corr = data.frame(Date = dates[1:length(dates)])
cGARCH_corr = cbind(cGARCH_corr, as.data.frame(a, col.names = colnames(pearson_corr)[2:(n_asset*n_asset+1)]))
colnames(cGARCH_corr) = colnames(pearson_corr)

cGARCH_std = cbind(data.frame(Date = dates[1:length(dates)]), as.data.frame(sigma(fit1)))
cGARCH_std = data.frame(cGARCH_std, row.names = NULL)

melted_data <- melt(cGARCH_corr[,c('Date','SP500 & STOXX',
                                   'SP500 & HSI',
                                   'SP500 & BOVESPA',
                                   'SP500 & NIFTY50')], id.vars = "Date")
melted_data$Date = as.Date(melted_data$Date)
ggplot(melted_data, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  labs(title = "cGARCH Correlation",
       x = "Date",
       y = "Value",
       color = "Variable") +
  theme_minimal()+
  theme(legend.position = "top")


# GJR-GARCH and A-DCC

# Garch_Models = c("sGARCH", "eGARCH", "gjrGARCH", "iGARCH")
# 
# for (j in 1:ncol(returns)) {
#   AIC = c()
#   for (i in 1:length(Garch_Models)) {
#     garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
#                               variance.model = list(garchOrder = c(1,1), 
#                                                     model = Garch_Models[i]), 
#                               distribution.model = "norm")
#     garch = ugarchfit(spec = garch11.spec, 
#                       data = na.omit(returns[,j]), solver.control = list(trace=0))
#     AIC = c(AIC,paste(Garch_Models[i],":",infocriteria(garch)[1]))
#   }
#   print(paste(colnames(returns)[j], AIC))
# }


gjrgarch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                             variance.model = list(garchOrder = c(1,1), 
                                                   model = "apARCH"), 
                             distribution.model = "norm")
adcc.garch11.spec = dccspec(uspec = multispec(replicate(n_asset, gjrgarch11.spec) ), 
                            dccOrder = c(1,1), 
                            distribution = "mvnorm",
                            model = 'aDCC')
adcc.garch11.spec
adcc.fit = dccfit(dcc.garch11.spec, data = na.omit(returns))

a = aperm(rcor(adcc.fit), c(3,1,2))
a = matrix(a, ncol = n_asset*n_asset)
aDCC_corr = data.frame(Date = dates[1:length(dates)])
aDCC_corr = cbind(aDCC_corr, as.data.frame(a, col.names = colnames(pearson_corr)[2:(n_asset*n_asset+1)]))
colnames(aDCC_corr) = colnames(pearson_corr)

melted_data <- melt(aDCC_corr[,c('Date','SP500 & STOXX',
                                 'SP500 & HSI',
                                 'SP500 & BOVESPA',
                                 'SP500 & NIFTY50')], id.vars = "Date")
melted_data$Date = as.Date(melted_data$Date)
ggplot(melted_data, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  labs(title = "aDCC Correlation",
       x = "Date",
       y = "Value",
       color = "Variable") +
  theme_minimal()+
  theme(legend.position = "top")


aDCC_std = cbind(data.frame(Date = dates[1:length(dates)]), as.data.frame(sigma(adcc.fit)))
aDCC_Std = data.frame(aDCC_std, row.names = NULL)

a = data.frame(cbind(DCC_corr$Date[120:nrow(DCC_corr)],
                     pearson_corr$`SP500 & STOXX`,
                     DCC_corr$`SP500 & STOXX`[120:nrow(DCC_corr)],
                     cGARCH_corr$`SP500 & STOXX`[120:nrow(DCC_corr)],
                     aDCC_corr$`SP500 & STOXX`[120:nrow(DCC_corr)]))
colnames(a) = c('Date','Pearson','DCC','CGARCH','aDCC')
melted_data <- melt(a, id.vars = "Date")
melted_data$value = as.numeric(melted_data$value)
melted_data$Date = as.Date(melted_data$Date)
ggplot(melted_data, aes(x = Date, y = value, color = variable)) +
  geom_line() +
  labs(title = "Correlation",
       x = "Date",
       y = "Value",
       color = "Variable") +
  theme_minimal()+
  theme(legend.position = "top")

# write.csv(aDCC_std, file = 'GJR GARCH std.csv')

write.csv(pearson_std, file = 'Pearson_std_stocks.csv')
write.csv(pearson_corr, file = 'Pearson_corr_stocks.csv')

write.csv(DCC_std, file = 'DCC_std_stocks.csv')
write.csv(DCC_corr, file = 'DCC_corr_stocks.csv')

write.csv(cGARCH_std, file = 'cGARCH_std_stocks.csv')
write.csv(cGARCH_corr, file = 'cGARCH_corr_stocks.csv')

write.csv(aDCC_std, file = 'ADCC_std_stocks.csv')
write.csv(aDCC_corr, file = 'ADCC_corr_stocks.csv')


#MS GARCH
MSGARCH_std = DCC_Std

spec = CreateSpec( variance.spec=list(model=c("sGARCH","gjrGARCH")), 
                   distribution.spec=list(distribution=c("norm","norm")), 
                   switch.spec=list(do.mix =FALSE,K=NULL))

msgarch_model = MSGARCH::FitML(spec=spec,data=returns$BOVESPA)
MSGARCH_std$BOVESPA = Volatility(msgarch_model)

msgarch_model = MSGARCH::FitML(spec=spec,data=returns$HSI)
MSGARCH_std$HSI = Volatility(msgarch_model)

msgarch_model = MSGARCH::FitML(spec=spec,data=returns$NIFTY50)
MSGARCH_std$NIFTY50 = Volatility(msgarch_model)

msgarch_model = MSGARCH::FitML(spec=spec,data=returns$STOXX)
MSGARCH_std$STOXX = Volatility(msgarch_model)

write.csv(MSGARCH_std, file = 'MSGARCH_std_stocks.csv')




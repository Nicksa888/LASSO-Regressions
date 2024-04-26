#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all data sets, functions and so on.

####################
####################
# Source Documents #
####################
####################

source("C:/R Portfolio/Global_Terrorism_Prediction/Functions 29 06 23.R")
source("C:/R Portfolio/Global_Terrorism_Prediction/MENA_Data_Object_Locations_13_04_2023.R")

# Data Preparation

# Here we select the Middle East and North Africa Region fro 

# Middle East & North Africa ---------------------------------------------

Region_Name <- "Middle East & North Africa"
MENA <- Region_Prep(GTD_WD, Region_Name)
glimpse(MENA)
MENA$Lethal <- as.integer(MENA$Lethal)
MENA$Lethal[MENA$Lethal %in% "1"] <- "0"
MENA$Lethal[MENA$Lethal %in% "2"] <- "1"
MENA$Lethal <- as.integer(MENA$Lethal)

####################
####################
# One Hot Encoding # 
####################
####################

# Here we convert the categorical variables into binary variables

MENA_Binary <- one_hot(as.data.table(MENA))
glimpse(MENA_Binary)

# Recode some variables #

MENA_Binary <- MENA_Binary %>% 
  rename(
    Islamic_State = `Group_Islamic State of Iraq and the Levant (ISIL)`,
    OtherGroup = Group_OtherGroup,
    Iraq = Country_Iraq,
    Syria = Country_Syria,
    Turkey = Country_Turkey,
    Yemen = Country_Yemen,
    OtherCountry = Country_OtherCountry,
    Iraq_Nationality = Nationality_Iraq,
    Israel_Nationality = Nationality_Israel,
    Turkey_Nationality = Nationality_Turkey,
    Yemen_Nationality = Nationality_Yemen,
    OtherNationality = Nationality_OtherNationality,
    Al_Anbar_Province = Province_Al_Anbar,
    Baghdad_Province = Province_Baghdad,
    Diyala_Province = Province_Diyala,
    Nineveh_Province = Province_Nineveh,
    Saladin_Province = Province_Saladin,
    OtherProvince = Province_OtherProvince,
    Baghdad_City = City_Baghdad,
    OtherCity = City_OtherCity
  )

# Remove unneeded parts of column names

names(MENA_Binary) = gsub(pattern = "Quarter_", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Week", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Group_", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Target_",
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Attack_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Weapon_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Nationality_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Province_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "City_",
                          replacement = "",
                          x = names(MENA_Binary))

write.csv(MENA_Binary, file = "MENA_Binary.csv", row.names = F)
MENA_Binary <- read.csv("MENA_Binary.csv")

# Final Prediction Data Set #

MENA_Binary_Initial <- dplyr::select(MENA_Binary, -c(all_of(Intial_Columns_Remove)))
MENA_Binary_Initial <- transform(MENA_Binary_Initial, Lethal = as.numeric(Lethal))

# Count NA's

sapply(MENA_Binary_Initial, function(x) sum(is.na(x)))
# Remove NA values

MENA_Binary_Initial <- MENA_Binary_Initial[complete.cases(MENA_Binary_Initial), ]

# zero and near - zero variance features #

# Zero and near-zero variance features refer to variables in a dataset that have either zero or very little variance. These features typically do not provide useful information for predictive modeling and may even hinder the performance of the model. Here's an explanation of both:

# Zero Variance Features:
# These are variables that have the same value for all observations in the dataset. Because the variable does not vary at all, it cannot help differentiate between different observations.
# For example, if a variable has a value of 0 for all observations, it has zero variance.
# Near-Zero Variance Features:
# These are variables that have very little variation in their values across the dataset. While they may have more than one unique value, one value might dominate the majority of the observations, making it nearly constant.
# Near-zero variance features are often undesirable because they don't provide enough information to the model and can lead to overfitting.
# Identifying and removing near-zero variance features can help simplify the model and improve its performance.
# In predictive modeling, it's essential to identify and handle zero and near-zero variance features appropriately. Removing these features can streamline the model, reduce complexity, and improve its ability to generalize to new data. Techniques such as univariate analysis, variance thresholds, and correlation analysis are commonly used to detect and address these issues.

set.seed(555)
feature_variance <- caret::nearZeroVar(MENA_Binary_Initial, saveMetrics = T)
head(feature_variance)
which(feature_variance$zeroVar == 'TRUE')

# There is no near zero or zero variance

# Correlation Test on Runs Train Data #

MENA_Binary_Initial_corr <- cor(MENA_Binary_Initial, method = "spearman")

high_corr_MENA_Binary_Initial <- caret::findCorrelation(MENA_Binary_Initial_corr, cutoff = 0.70)
high_corr_MENA_Binary_Initial

# MENA_Binary_Initial_corr <- cor(MENA_Binary_Initial, method = "spearman"):
#   This line calculates the Spearman correlation coefficient matrix for the dataset MENA_Binary_Initial.
# The method = "spearman" argument specifies that Spearman's rank correlation coefficient is used. Spearman correlation measures the strength and direction of association between two variables, and it's particularly useful for assessing monotonic relationships between variables.
# high_corr_MENA_Binary_Initial <- caret::findCorrelation(MENA_Binary_Initial_corr, cutoff = 0.70):
#   This line applies the findCorrelation function from the caret package to identify highly correlated variables in the correlation matrix MENA_Binary_Initial_corr.
# The cutoff = 0.70 argument specifies the threshold for correlation. Here, any pair of variables with a correlation coefficient greater than or equal to 0.70 (in absolute value) will be considered highly correlated.
# The function returns a logical vector indicating which variables are highly correlated with each other. If two variables are highly correlated, one of them might be redundant for modeling purposes, and removing one can help avoid multicollinearity issues.

# These are the variables that are highlu correlated

#  18 23 29 34 12 15 21  3 20
names(MENA_Binary_Initial)

# "Iraq", "Iraq_Nationality", "Baghdad_Province", "Baghdad_City", "BombAttack", "Explosives", "Yemen", "Islamic_State", "Turkey" 

MENA_Binary_Final <- dplyr::select(MENA_Binary_Initial, -c(all_of(Corr_Columns_Remove)))

# Predict for 2020

# For Training Data, select all data upto and including 2019.
MENA_Train_Year <- MENA_Binary_Final %>% dplyr::filter(Year %in% c(1970:2019)) %>% dplyr::select(-c(Year))
MENA_Test_Year <- dplyr::filter(MENA_Binary_Final, Year == 2020) %>% dplyr::select(-c(Year))
MENA_Train_Year <- transform(MENA_Train_Year, Lethal = as.numeric(Lethal))
MENA_Test_Year <- transform(MENA_Test_Year, Lethal = as.numeric(Lethal))

###################
# Lasso Modelling #
###################

# define response variable

MENA_Train_Year$Lethal <- as.numeric(MENA_Train_Year$Lethal)
y_Year <- MENA_Train_Year$Lethal

# define matrix of predictor variables

MENA_Train_Year <- as.data.frame(MENA_Train_Year)
names(MENA_Train_Year)
MENA_Year_Names <- c("OtherGroup", 
                     "Business",           
                     "GovtGen",
                     "OtherTarget", 
                     "Police",             
                     "Private", 
                     "ArmedAssaultAttack",
                     "Assassination",      
                     "HostageKidnapAttack",
                     "OtherAttack",  
                     "Firearms",           
                     "OtherWeapon",   
                     "OtherCountry", 
                     "Syria",              
                     "Israel_Nationality", 
                     "OtherNationality",  
                     "Turkey_Nationality", 
                     "Yemen_Nationality", 
                     "Al_Anbar_Province",  
                     "Diyala_Province",    
                     "Nineveh_Province",  
                     "OtherProvince", 
                     "Saladin_Province",   
                     "OtherCity")
MENA_Train_year <- subset(MENA_Train_Year, select = -Lethal)
x_Year_Lasso <- data.matrix(MENA_Train_year[MENA_Year_Names])
x_Year_Lasso
names(x_Year_Lasso)

# Regular LASSO Modelling

#perform k-fold cross-validation to find optimal lambda value
cv_model_Year_Lasso <- cv.glmnet(x_Year_Lasso, y_Year, alpha = 1, nfolds = 10)

# This code snippet above performs k-fold cross-validation for the Lasso regression model using the cv.glmnet function from the glmnet package. Here's a breakdown of each component:
# 
# cv_model_Year_Lasso: This variable stores the result of the cross-validated Lasso regression model.
# cv.glmnet(): This function conducts cross-validation for a Lasso (or elastic net) regression model. It takes several arguments:
# x_Year_Lasso: The predictor variables (in matrix form) for the model.
# y_Year: The response variable (dependent variable) for the model.
# alpha = 1: Specifies that Lasso regularization is used. An alpha value of 1 corresponds to the Lasso penalty, while an alpha value of 0 would correspond to the ridge penalty.
# nfolds = 10: Specifies the number of folds for cross-validation. In this case, it performs 10-fold cross-validation, meaning the data is divided into 10 subsets, and the model is trained and tested 10 times, each time using a different subset as the test set.
# Cross-validation is a technique used to assess how well a predictive model will generalize to an independent dataset. In k-fold cross-validation, the dataset is randomly partitioned into k equal-sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results are then averaged to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.

#find optimal lambda value that minimizes test MSE
best_lambda_Year_Lasso <- cv_model_Year_Lasso$lambda.min
best_lambda_Year_Lasso

# This code above retrieves the optimal value of the regularization parameter lambda selected by the cross-validated Lasso regression model.
# 
# cv_model_Year_Lasso$lambda.min: This part accesses the lambda.min component of the cv_model_Year_Lasso object. The cv_model_Year_Lasso object stores the results of cross-validation performed on the Lasso regression model. The lambda.min component represents the value of lambda that minimizes the cross-validated error. This lambda value is determined during the cross-validation process as the optimal regularization parameter for the model.
# best_lambda_Year_Lasso: This variable stores the value of the optimal lambda parameter, which is selected based on the cross-validation results. This lambda value is then used to fit the final Lasso regression model.

# An Improved LASSO Model

# This code fits a Lasso regression model to the training data using the optimal value of the regularization parameter lambda determined through cross-validation.
# 
# glmnet: This function is used to fit a generalized linear model via penalized maximum likelihood. It is specifically designed for fitting regularized regression models, such as Lasso and Ridge regression.
# x_Year_Lasso: This is the matrix of predictor variables used to train the model.
# y_Year: This is the response variable used to train the model.
# alpha = 1: The alpha parameter specifies the type of penalty term used in the model. In this case, alpha = 1 indicates that the Lasso penalty (L1 regularization) is applied.
# nfolds = 10: This parameter specifies the number of folds used in cross-validation. The data is divided into 10 folds, and the model is trained and tested iteratively on different combinations of folds.
# lambda = best_lambda_Year_Lasso: This parameter specifies the value of the regularization parameter lambda to be used in the model. best_lambda_Year_Lasso is the optimal value of lambda selected through cross-validation, which minimizes the cross-validated error.
# # a better version of LASSO: This comment indicates that the Lasso regression model with the optimal lambda value selected through cross-validation is expected to perform better in terms of predictive accuracy compared to other versions of the Lasso model with different lambda values.

best_model_Year_Lasso <- glmnet(x_Year_Lasso, 
                                y_Year, 
                                alpha = 1, 
                                nfolds = 10, 
                                lambda = best_lambda_Year_Lasso) # a better version of LASSO


coef(best_model_Year_Lasso)

# Extract the coefficients
coefficients <- coef(best_model_Year_Lasso)

# Create the algebraic formula
formula_string <- paste("Lethal =", as.character(coefficients[1]), " * (Intercept)")

# Loop through each predictor variable
for (i in 1:length(MENA_Year_Names)) {
  # Check if the coefficient is not zero
  if (coefficients[i + 1] != 0) {
    # Append the term to the formula string
    formula_string <- paste(formula_string, "+", as.character(coefficients[i + 1]), "*", MENA_Year_Names[i])
  }
}

# Print the algebraic formula
print(formula_string)

Lasso_Year_Regression <- glm(Lethal ~ OtherGroup +               
                               Business +                 
                               GovtGen +                 
                               Police +                   
                               Private +                 
                               OtherTarget +          
                               ArmedAssaultAttack +      
                               Assassination +
                               HostageKidnapAttack +
                               OtherAttack +             
                               Firearms +                 
                               OtherWeapon +  
                               Syria +
                               Israel_Nationality +   
                               OtherNationality +    
                               Turkey_Nationality +  
                               Yemen_Nationality +   
                               Al_Anbar_Province +   
                               Diyala_Province +      
                               Nineveh_Province +    
                               OtherProvince +         
                               OtherCity, 
                               family = binomial(link = "logit"), MENA_Train_Year)

#########################
# Perform Relaxed Lasso #
#########################

# Here's an explanation of the code:
# 
# glmnet: As before, this function is used to fit a generalized linear model via penalized maximum likelihood, specifically for regularized regression models like Lasso.
# x_Year_Lasso: Matrix of predictor variables used to train the model.
# y_Year: Response variable used to train the model.
# alpha = 1: Specifies that the Lasso penalty (L1 regularization) is applied.
# nfolds = 10: Specifies the number of folds used in cross-validation.
# lambda = best_lambda_Year_Lasso: Specifies the value of the regularization parameter lambda to be used in the model. best_lambda_Year_Lasso is the optimal lambda value selected through cross-validation, which minimizes the cross-validated error.
# relax = TRUE: This additional argument tells the glmnet function to relax the Lasso penalty for certain variables. Relaxing the penalty means that these variables are treated less strictly in terms of regularization, allowing them to have larger coefficients in the model. This can be useful when dealing with multicollinearity or when there are groups of highly correlated predictors. By relaxing the penalty for some variables, the model may achieve better predictive performance.

best_model_Year_Lasso_relaxed <- glmnet(x_Year_Lasso, 
                                        y_Year, 
                                        alpha = 1, 
                                        nfolds = 10, 
                                        lambda = best_lambda_Year_Lasso,
                                        relax = TRUE) # a better version of LASSO

##########################
# Perform adaptive LASSO #
##########################

## The intercept estimate should be dropped.
best_ridge_coef <- as.numeric(coef(cv_model_Year_Lasso, 
                                   best_lambda_Year_Lasso <- cv_model_Year_Lasso$lambda.min))[-1]

best_model_Year_Lasso_adaptive <- cv.glmnet(
             x = x_Year_Lasso, 
             y = y_Year,
             alpha = 1,
             penalty.factor = 1 / abs(best_ridge_coef))
best_model_Year_Lasso_adaptive
coef(best_model_Year_Lasso_adaptive)

# # This code snippet above is performing adaptive LASSO regularization using cross-validation to find the optimal penalty parameter (lambda). Here's an explanation of the parameters:
# 
# x: The matrix of predictor variables.
# y: The response variable.
# alpha: The elastic net mixing parameter. When alpha = 1, it represents the LASSO penalty, which encourages sparsity in the coefficients.
# penalty.factor: A vector of length equal to the number of predictors, where each element represents a separate penalty factor that can be applied to each coefficient. In adaptive LASSO, the penalty factors are adjusted based on the ridge regression coefficients (best_ridge_coef), with the goal of providing differential shrinkage to different variables. This means that some variables may be more heavily penalized than others, allowing the model to perform variable selection and regularization more effectively.
# Overall, this code is setting up a cross-validated adaptive LASSO regression model, where the regularization strength is adaptively adjusted for each predictor based on its estimated coefficients from ridge regression.

# ================================
# STEP 0: INSTALL + LOAD PACKAGES
# ================================

required_packages <- c(
  "tidyverse",    # data manipulation + ggplot
  "caret",        # ML training + evaluation + sampling utilities
  "pROC",         # ROC + AUC
  "corrplot",     # correlation plot
  "e1071",        # SVM (caret uses it)
  "randomForest", # Random Forest
  "rpart",        # Decision Tree
  "rpart.plot"    # Tree plot
)

installed <- rownames(installed.packages())
for (p in required_packages) {
  if (!p %in% installed) install.packages(p, dependencies = TRUE)
}

library(tidyverse)
library(caret)
library(pROC)
library(corrplot)
library(e1071)
library(randomForest)
library(rpart)
library(rpart.plot)



# ================================
# STEP 1: SET WORKING DIRECTORY + LOAD DATA
# ================================

data <- read.csv("D:/creditcard.csv")

# Basic checks
dim(data)
names(data)
str(data)
summary(data)





# ================================
# STEP 2: DATA QUALITY CHECKS
# ================================

# Missing values
colSums(is.na(data))

# Duplicates (optional check)
sum(duplicated(data))

# Class distribution
table(data$Class)
prop.table(table(data$Class))






# ================================
# STEP 3: PREPROCESSING
# ================================

data$Class <- factor(data$Class, levels = c(0, 1), labels = c("NonFraud", "Fraud"))

# Scale Time and Amount (keep original too if you want)
data$Time_Scaled <- as.numeric(scale(data$Time))
data$Amount_Scaled <- as.numeric(scale(data$Amount))

# Drop original Time and Amount (optional but cleaner for modelling)
data_model <- data %>%
  select(-Time, -Amount)

# Confirm structure
str(data_model)







# ================================
# STEP 4: EDA (PLOTS FOR REPORT)
# ================================

# Create folder for figures
if (!dir.exists("figures")) dir.create("figures")

# 4.1 Class distribution plot
p_class <- ggplot(data_model, aes(x = Class)) +
  geom_bar() +
  labs(title = "Class Distribution (Fraud vs Non-Fraud)", x = "Class", y = "Count")
p_class
ggsave("figures/class_distribution.png", p_class, width = 7, height = 5)

# 4.2 Amount distribution by Class (use original data for interpretability)
p_amount <- ggplot(data, aes(x = Amount, fill = factor(Class))) +
  geom_histogram(bins = 50) +
  labs(title = "Transaction Amount Distribution by Class",
       x = "Amount", y = "Frequency", fill = "Class")
p_amount
ggsave("figures/amount_distribution.png", p_amount, width = 7, height = 5)

# 4.3 Boxplot of Amount by Class
p_amount_box <- ggplot(data, aes(x = factor(Class), y = Amount)) +
  geom_boxplot() +
  labs(title = "Amount by Class (Boxplot)", x = "Class", y = "Amount")
p_amount_box
ggsave("figures/amount_boxplot.png", p_amount_box, width = 7, height = 5)

# 4.4 Time trend sample (optional) - useful to show pattern
# For large dataset, plotting all points is heavy, so sample
set.seed(123)
sample_idx <- sample(nrow(data), size = min(20000, nrow(data)))
data_sample <- data[sample_idx, ]

p_time <- ggplot(data_sample, aes(x = Time, y = Amount, color = factor(Class))) +
  geom_point(alpha = 0.5) +
  labs(title = "Time vs Amount (Sampled)", x = "Time", y = "Amount", color = "Class")
p_time
ggsave("figures/time_vs_amount_sample.png", p_time, width = 7, height = 5)

# 4.5 Correlation plot (sample to avoid memory issues)
# Use only numeric columns excluding Class
num_cols <- data_model %>%
  select(-Class) %>%
  select(where(is.numeric))

set.seed(123)
corr_sample <- num_cols[sample(1:nrow(num_cols), size = min(30000, nrow(num_cols))), ]
corr_matrix <- cor(corr_sample)

png("figures/correlation_matrix.png", width = 1000, height = 800)
corrplot(corr_matrix, method = "color", type = "lower", tl.cex = 0.6)
dev.off()






# ================================
# STEP 5: TRAIN-TEST SPLIT (STRATIFIED)
# ================================

set.seed(123)

train_index <- createDataPartition(data_model$Class, p = 0.70, list = FALSE)
train_data <- data_model[train_index, ]
test_data  <- data_model[-train_index, ]

table(train_data$Class)
table(test_data$Class)
prop.table(table(train_data$Class))
prop.table(table(test_data$Class))






# ================================
# STEP 6: HANDLE IMBALANCE (DOWNSAMPLING TRAIN SET)
# ================================

set.seed(123)

train_down <- downSample(
  x = train_data %>% select(-Class),
  y = train_data$Class,
  yname = "Class"
)

table(train_down$Class)
prop.table(table(train_down$Class))






# ================================
# STEP 7: TRAIN CLASSIFICATION MODELS
# ================================

set.seed(123)

# Training control: repeated CV
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 7.1 Logistic Regression
model_lr <- train(
  Class ~ .,
  data = train_down,
  method = "glm",
  family = binomial(),
  trControl = ctrl,
  metric = "ROC"
)

# 7.2 Decision Tree (rpart)
model_dt <- train(
  Class ~ .,
  data = train_down,
  method = "rpart",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10
)

# 7.3 Random Forest
model_rf <- train(
  Class ~ .,
  data = train_down,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 3
)

# 7.4 SVM Radial
model_svm <- train(
  Class ~ .,
  data = train_down,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)

# Compare CV performance
model_lr
model_dt
model_rf
model_svm







# ================================
# STEP 8: EVALUATION ON TEST SET
# ================================

# Function: Evaluate model with confusion matrix + ROC/AUC
evaluate_model <- function(model, test_df, model_name) {
  # Class predictions
  pred_class <- predict(model, newdata = test_df)
  
  # Probabilities for ROC (Fraud class prob)
  pred_prob <- predict(model, newdata = test_df, type = "prob")[, "Fraud"]
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, test_df$Class, positive = "Fraud")
  
  # ROC + AUC
  roc_obj <- roc(response = test_df$Class, predictor = pred_prob, levels = c("NonFraud", "Fraud"))
  auc_val <- auc(roc_obj)
  
  list(
    name = model_name,
    confusion = cm,
    roc = roc_obj,
    auc = auc_val
  )
}

res_lr  <- evaluate_model(model_lr,  test_data, "Logistic Regression")
res_dt  <- evaluate_model(model_dt,  test_data, "Decision Tree")
res_rf  <- evaluate_model(model_rf,  test_data, "Random Forest")
res_svm <- evaluate_model(model_svm, test_data, "SVM Radial")

# Print key results
res_lr$confusion
res_dt$confusion
res_rf$confusion
res_svm$confusion

res_lr$auc
res_dt$auc
res_rf$auc
res_svm$auc





# ================================
# STEP 9: ROC CURVES COMPARISON PLOT
# ================================

png("figures/roc_comparison.png", width = 900, height = 700)

plot(res_lr$roc, main = "ROC Curve Comparison (Test Set)")
plot(res_dt$roc, add = TRUE)
plot(res_rf$roc, add = TRUE)
plot(res_svm$roc, add = TRUE)

legend(
  "bottomright",
  legend = c(
    paste0(res_lr$name,  " AUC=", round(res_lr$auc, 4)),
    paste0(res_dt$name,  " AUC=", round(res_dt$auc, 4)),
    paste0(res_rf$name,  " AUC=", round(res_rf$auc, 4)),
    paste0(res_svm$name, " AUC=", round(res_svm$auc, 4))
  ),
  lwd = 2,
  cex = 0.9
)

dev.off()






# ================================
# STEP 10: EXPORT SUMMARY RESULTS TABLE
# ================================

get_metrics <- function(cm_obj, auc_val, model_name) {
  acc <- cm_obj$overall["Accuracy"]
  prec <- cm_obj$byClass["Precision"]
  rec <- cm_obj$byClass["Recall"]
  f1 <- cm_obj$byClass["F1"]
  
  data.frame(
    Model = model_name,
    Accuracy = as.numeric(acc),
    Precision = as.numeric(prec),
    Recall = as.numeric(rec),
    F1 = as.numeric(f1),
    AUC = as.numeric(auc_val)
  )
}

results_table <- bind_rows(
  get_metrics(res_lr$confusion,  res_lr$auc,  res_lr$name),
  get_metrics(res_dt$confusion,  res_dt$auc,  res_dt$name),
  get_metrics(res_rf$confusion,  res_rf$auc,  res_rf$name),
  get_metrics(res_svm$confusion, res_svm$auc, res_svm$name)
)

results_table

write.csv(results_table, "model_results_summary.csv", row.names = FALSE)



# ================================
# STEP 11: SAVE TRAINED MODELS (OPTIONAL)
# ================================

saveRDS(model_lr,  "model_logistic_regression.rds")
saveRDS(model_dt,  "model_decision_tree.rds")
saveRDS(model_rf,  "model_random_forest.rds")
saveRDS(model_svm, "model_svm_radial.rds")



# ================================
# STEP 12: SESSION INFO
# ================================
sessionInfo()




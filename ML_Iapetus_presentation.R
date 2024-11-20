# If needed
# install.packages('tidyverse','tidymodels', 'modeldata', 'glmnet', 'parallel', 'ranger', 'vip')

library('tidyverse')
library('tidymodels')
library('modeldata')
library('glmnet')
library('parallel')
library('ranger')
library('vip')

##----
urchins <-
  read_csv("https://tidymodels.org/start/models/urchins.csv") %>%
  setNames(c("food_regime", "initial_volume", "width")) %>%
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))

glimpse(urchins)


## ----
ggplot(urchins,aes(initial_volume, width, colour = food_regime)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE)


## ----
mod <- lm(width ~ initial_volume * food_regime, data = urchins)
summary (mod)


## ----
# Set up the model type and engine
lin_mod <- linear_reg() %>% set_engine ('glm')

# Fit the model
lin_fit <- lin_mod %>% fit(width ~ initial_volume * food_regime,
                           data = urchins,
                           family='gaussian')

# From the broom pkg
tidy(lin_fit)



## ----
hotels <-
  read_csv("https://tidymodels.org/start/case-study/hotels.csv") %>%
  mutate(across(where(is.character), as.factor))

glimpse(hotels)


## ----
# Do we have missing data?
hotels %>% tally()
hotels %>% na.omit() %>% tally()

# What does the outcome variable look like?
hotels %>%
  count(children) %>%
  mutate(prop = n/sum(n))



## ----
set.seed(45)

# Stratified sampling by the children variable to keep proportions
data_split <- initial_split(hotels, prop = 3/4, strata = children)

hotel_train <- training(data_split)
hotel_test <- testing(data_split)

print(c(nrow(hotels),nrow(hotel_train),nrow(hotel_test)))

hotel_train %>%
  count(children) %>%
  mutate(prop = n/sum(n))

hotel_test %>%
  count(children) %>%
  mutate(prop = n/sum(n))


## ---------------------------------------------------------------------------------------------------------
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter",
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <-
  recipe(children ~ ., data = hotel_train) %>% # model firmula
  step_date(arrival_date) %>% # splits dates into y, m, d
  step_holiday(arrival_date, holidays = holidays) %>% # holiday dummy vars
  step_rm(arrival_date) %>% # remove variables
  step_dummy(all_nominal_predictors()) %>%  # encodes chr or fct to dummies
  step_zv(all_predictors()) %>% # remove variables with zero variance
  step_normalize(all_predictors()) # centers and scales


## ---------------------------------------------------------------------------------------------------------
# Set up the model
# tune() indicates this parameters should be tuned
lr_mod <-
  logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")


## ---------------------------------------------------------------------------------------------------------
lr_workflow <-
  workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(lr_recipe)


## ---------------------------------------------------------------------------------------------------------
# Create three folds
set.seed(234)
hotel_folds <- vfold_cv(hotel_train,v=3,strata= children)

hotel_folds


## ---------------------------------------------------------------------------------------------------------
# Create a regular sequence for the penalty parameter
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
nrow(lr_reg_grid)
head(lr_reg_grid)


## ---------------------------------------------------------------------------------------------------------
lr_res <-
  lr_workflow %>%
  tune_grid(resamples = hotel_folds, # validation data
            grid = lr_reg_grid,# hyperparameter values
            control = control_grid(save_pred = TRUE), # save the cval results
            metrics = metric_set(roc_auc)) # which performance metric to use


## ---------------------------------------------------------------------------------------------------------
lr_perf <- lr_res %>% collect_metrics()

lr_perf


## ---------------------------------------------------------------------------------------------------------
lr_perf %>%
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() +
  geom_line() +
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())


## ---------------------------------------------------------------------------------------------------------
top_models <-
  lr_res %>%
  show_best(metric = "roc_auc", n = 15) %>%
  arrange(desc(mean))
top_models


## ---------------------------------------------------------------------------------------------------------
lr_best <- lr_res %>% select_best(metric = 'roc_auc')

lr_best



## ---------------------------------------------------------------------------------------------------------
lr_auc <-
  lr_res %>%
  collect_predictions(parameters = lr_best) %>%
  roc_curve(children, .pred_children) %>%
  mutate(model = "Logistic Regression")

autoplot(lr_auc)


## ---------------------------------------------------------------------------------------------------------
cores <- parallel::detectCores()
cores


## ---------------------------------------------------------------------------------------------------------
rf_mod <-
  rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine("ranger", num.threads = cores-2) %>%
  set_mode("classification")


## ---------------------------------------------------------------------------------------------------------
rf_recipe <-
  recipe(children ~ ., data = hotel_train) %>% # model firmula
  step_date(arrival_date) %>% # splits dates into y, m, d
  step_holiday(arrival_date, holidays = holidays) %>% # holiday dummy vars
  step_rm(arrival_date) # remove variables


## ---------------------------------------------------------------------------------------------------------
rf_workflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)


## ---------------------------------------------------------------------------------------------------------
rf_grid <- expand.grid(
      mtry = c(3, 5, 7),
      trees = c(500, 1000, 5000))

rf_grid


## ---------------------------------------------------------------------------------------------------------
rf_res <-
  rf_workflow %>%
  tune_grid(hotel_folds,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


## ---------------------------------------------------------------------------------------------------------
rf_res %>%
  show_best(metric = "roc_auc")

autoplot(rf_res)

rf_best <-
  rf_res %>%
  select_best(metric = "roc_auc")


## ---------------------------------------------------------------------------------------------------------
rf_auc <-
  rf_res %>%
  collect_predictions(parameters = rf_best) %>%
  roc_curve(children, .pred_children) %>%
  mutate(model = "Random Forest")


## ---------------------------------------------------------------------------------------------------------
bind_rows(rf_auc, lr_auc) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() +
  scale_color_viridis_d(option = "plasma", end = .6)


## ---------------------------------------------------------------------------------------------------------
# the last model
last_rf_mod <-
  rand_forest(mtry = 5, trees = 1000) %>%
  set_engine("ranger", num.threads = cores-2, importance = "impurity") %>%
  set_mode("classification")

# the last workflow
last_rf_workflow <-
  rf_workflow %>%
  update_model(last_rf_mod)

# the last fit
set.seed(345)
last_rf_fit <-
  last_rf_workflow %>%
  last_fit(data_split)

last_rf_fit


## ---------------------------------------------------------------------------------------------------------
last_rf_fit %>%
  collect_metrics()


## ---------------------------------------------------------------------------------------------------------
library(vip)
last_rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20)


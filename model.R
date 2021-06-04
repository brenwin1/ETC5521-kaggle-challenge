# ==================== load libraries ====================
library(tidyverse)
library(tidymodels)
library(rsample)
library(recipes)
library(tune)
library(dials)



# ==================== read in data + clean script ====================
# --- read in training data fires_tr_full
fires_tr_full <- read_csv("data/student_training_release.csv") %>% 
  mutate(cause = factor(cause,
                        levels = c("lightning", "accident", "arson", "burning_off"))) %>% 
  na.omit() %>% # omit missing observations 
  filter(month %in% c(1:3, 10:12)) # filter to months in test set

# --- read in test set fires_to_predict
fires_to_predict <- read_csv("data/student_predict_x_release.csv") %>% 
  select(-id) # remove id column

# impute most common factor level; to 1 missing obs.
fires_to_predict <- fires_to_predict %>% 
  mutate(FOR_TYPE = case_when(is.na(FOR_TYPE) ~ "Non forest",
                              TRUE ~ FOR_TYPE),
         FOR_CAT = case_when(is.na(FOR_CAT) ~ "Native forest",
                             TRUE ~ FOR_CAT),
         COVER = case_when(is.na(COVER) ~ 2,
                           TRUE ~ COVER),
         HEIGHT = case_when(is.na(HEIGHT) ~ 2,
                            TRUE ~ HEIGHT),
         FOREST = case_when(is.na(FOREST) ~ 1,
                            TRUE ~ FOREST)) 

# add cause variable; so can run recipe on test set
fires_to_predict <- fires_to_predict %>% 
  mutate(cause = fires_tr_full$cause[c(1:1022)]) %>% 
  mutate(cause = factor(cause,
                        levels = c("lightning", "accident", "arson", "burning_off")))



# ==================== Data pre-processing (recipe) ====================

# --- select predictors for model fitting

# training set
fires_tr_full <- fires_tr_full %>% 
  select(cause,
         month, lat, lon,  # spatio-temporal variables
         dist_cfa, dist_road, dist_camp, # distance variables
         arf720, ase180, mint, amaxt60, aws_m24) %>%  # climate variables
  mutate(month = factor(month))

# test set
fires_to_predict <- fires_to_predict %>% 
  select(cause,
         month, lat, lon,  # spatio-temporal variables
         dist_cfa, dist_road, dist_camp, # distance variables
         arf720, ase180, mint, amaxt60, aws_m24) %>%  # climate variables
  mutate(month = factor(month))

# ***** p/s for the 2nd best model; month is omitted

# --- split training set; into 2/3 training 1/3 test set via `rsample`
set.seed(3000)

split <- rsample::initial_split(fires_tr_full,
                                prop = 3/4,
                                strata = "cause") # ensure; cause; same proportion in training & test set

fires_tr <- rsample::training(split) # extract training set
fires_ts <- rsample::testing(split) # extract test set

# --- recipe
fires_rec <- recipes::recipe(cause ~ .,
                             data = fires_tr) %>% 
  step_log(dist_cfa, dist_camp, dist_road) %>% 
  step_dummy(month) # ***** month is omitted for 2nd best model

# --- prep & bake training set (apply recipe to training & test sets)
fires_tr_baked <- fires_rec %>% 
  recipes::prep(fires_tr) %>% 
  recipes::bake(fires_tr)

fires_tp_baked <- fires_rec %>% 
  recipes::prep(fires_to_predict) %>% 
  recipes::bake(fires_to_predict)



# ==================== Model specificiation & tuning ***hours to run this section (skip to next) ====================

# --- model specification x set computational engine x set hyperparameters to tune *NOTE: take few hours to run
tune_spec <- parsnip::rand_forest(mtry = tune(),
                                  trees = 1000,
                                  min_n = tune()) %>%
  set_engine("randomForest",  # set computational engine (randomForest package)
             importance = TRUE, # compute variable importance
             proximity = TRUE) %>% # compute proximity
  set_mode("classification") # set classification mode

# create workflow object (mesh recipe & model)
tune_wf <- workflow() %>% 
  add_recipe(fires_rec) %>% 
  add_model(tune_spec)

# create 10 fold cross-validation sets
set.seed(234)
fires_folds <- rsample::vfold_cv(fires_tr)

doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune::tune_grid(tune_wf, 
                            resamples = fires_folds,
                            grid = 20) 

# --- based on previous tuning results above create grid of `min_n` and `mtry` values; tune parameters on cv folds
rf_grid <- dials::grid_regular(dials::mtry(range = c(4, 15)),
                               dials::min_n(range = c(1, 10)),
                               levels = 6)

set.seed(456)
regular_res <- tune::tune_grid(tune_wf,
                               resamples = fires_folds,
                               grid = rf_grid)

# > therefore, the best model suggested by tuning has hyperparameters
# - `min_n` = 2
# - `mtry` = 1

# extract model; based on (highest) mean roc_auc metric *same as accuracy
best_auc <- tune::select_best(regular_res,
                              metric = "roc_auc")

# finalise model
final_rf <- tune::finalize_model(tune_spec,
                                 parameters = best_auc)

# finalise workflow (add recipe & final_rf model)
final_wf <- workflow() %>% 
  add_recipe(fires_rec) %>% 
  add_model(final_rf)

# apply pre-processing step & fit model
final_fit <- final_wf %>% 
  parsnip::fit(data = fires_tr) %>% 
  workflows::pull_workflow_fit() 

fires_to_predict_p <- fires_tp_baked %>% 
  mutate(pred_rf = predict(final_fit, fires_tp_baked)$.pred_class) %>% 
  mutate(Id = row_number()) %>% 
  select(Id, Category = pred_rf)

write_csv(fires_to_predict_p, file = "predictions_no1.csv")

# ==================== post tuning: model specification & set computational engine ====================
# *** to avoid teaching team having to tune; run this (specifies params. from tuning *** BUT STILL prone to different results from different seeds)
fires_rf1 <- parsnip::rand_forest(mtry = 4,
                                  trees = 1000,
                                  min_n = 1) %>% # model specification  
  set_engine("randomForest",  # set computational engine (randomForest package)
             importance = TRUE, # compute variable importance
             proximity = TRUE) %>% # compute proximity
  set_mode("classification") # set classification mode

# fit model 
fires_rf1 <- fires_rf1 %>% 
  parsnip::fit(cause ~ .,
               data = fires_tr_baked)

# use model; make predictions on test set
fires_to_predict_p <- fires_tp_baked %>% 
  mutate(pred_rf = predict(fires_rf1, 
                           fires_tp_baked)$.pred_class) %>% 
  mutate(Id = row_number(),
         Category = pred_rf) %>% 
  select(Id, Category)




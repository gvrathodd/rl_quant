File Descriptions
1. label.py
   • Computes regime labels (bull, bear, sideways, etc.) for historical price data.
   • Generates multiple horizon labels (e.g., h = 50, 100, ...) and saves them as CSV files.

2. base_models_4fold.py
   • Trains XGBoost models using a 4-fold chronological cross-validation setup.
   • Computes predicted probabilities for each class on test splits.

3. train_model.py
   • Uses precomputed labels to train XGBoost models.
   • Generates probabilities for test data and saves the results as CSV files.
   • Can be used to extend predictions for future days beyond the training period.

4. rl_model.py
   • Implements a reinforcement learning (RL) trading environment using gymnasium.
   • Trains a PPO agent based on predicted probabilities or price features.
   • Contains reward logic including transaction fees, holding bonuses, and penalties.

5. readme.txt
   • This file.
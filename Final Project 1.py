# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# %%

# -----------------------------------
# 1. Load dataset
# -----------------------------------
df = pd.read_csv("international-fifa-world-cup-2022-qatar-players-2022-to-2022-stats.csv")



# %%

print(df.shape)
print(df.head())

# %%
# Choose columns for the project
columns = [
    "goals_overall",
    "assists_overall",
    "minutes_played_overall",
    "shots_total_overall",
    "shots_on_target_overall",
    "yellow_cards_overall",
    "average_rating_overall"
]


# %%

# -----------------------------------
# 2. Select Important columns
# -----------------------------------
df = df[[
    "goals_overall",
    "assists_overall",
    "minutes_played_overall",
    "average_rating_overall"
]]

# %%

# -----------------------------------
# 3. Drop missing values
# -----------------------------------
df = df.dropna()
df = df.drop_duplicates()


# %%

# Remove Invalid values:
df = df[df["minutes_played_overall"] > 0]
df = df[df["average_rating_overall"] > 0]


# %%

# Remove unrelastic ratings:
df = df[df["average_rating_overall"] <= 10]

print("Dataset Cleaned", df.shape)

# %%

# -----------------------------------
# 4. Define features and target
# -----------------------------------
X = df[["goals_overall", "assists_overall", "minutes_played_overall"]]
y = df["average_rating_overall"]


# %%

# -----------------------------------
# 5. Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%

# -----------------------------------
# 6. Linear Regression
# -----------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5

print("Linear Regression")
print("R2:", r2_lr)
print("RMSE:", rmse_lr)
print()


# %%

# -----------------------------------
# 7. Decision Tree
# -----------------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = mean_squared_error(y_test, y_pred_dt) ** 0.5

print("Decision Tree")
print("R2:", r2_dt)
print("RMSE:", rmse_dt)
print()


# %%

# -----------------------------------
# 8. Random Forest
# -----------------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5

print("Random Forest")
print("R2:", r2_rf)
print("RMSE:", rmse_rf)
print()


# %%

# -----------------------------------
# 9. Results Table
# -----------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "R2": [r2_lr, r2_dt, r2_rf],
    "RMSE": [rmse_lr, rmse_dt, rmse_rf]
})

print("Model Comparison Table")
print(results)


# %%

# -----------------------------------
# 10. Bar Chart (R2)
# -----------------------------------
plt.figure()
plt.bar(results["Model"], results["R2"])
plt.title("Model Comparison (R2)")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.show()


# %%

# -----------------------------------
# 11. Bar Chart (RMSE)
# -----------------------------------
plt.figure()
plt.bar(results["Model"], results["RMSE"])
plt.title("Model Comparison (RMSE)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()


# %%

# -----------------------------------
# 12. Scatter Plots
# -----------------------------------

# Goals vs Rating
plt.figure()
plt.scatter(df["goals_overall"], df["average_rating_overall"])
plt.title("Goals vs Average Rating")
plt.xlabel("Goals")
plt.ylabel("Average Rating")
plt.show()


# %%

# Assists vs Rating
plt.figure()
plt.scatter(df["assists_overall"], df["average_rating_overall"])
plt.title("Assists vs Average Rating")
plt.xlabel("Assists")
plt.ylabel("Average Rating")
plt.show()


# %%

# Minutes vs Rating
plt.figure()
plt.scatter(df["minutes_played_overall"], df["average_rating_overall"])
plt.title("Minutes Played vs Average Rating")
plt.xlabel("Minutes Played")
plt.ylabel("Average Rating")
plt.show()



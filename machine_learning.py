# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def randomeForest_timeBar(df: pd.DataFrame) -> np.ndarray:
    """
    Predicts the close of bars using Random Forest.
    Args:
        df(pd.DataFrame): OHLCV Data
    Return:
        np.ndarray: Prediction of closing values using OHLCV data
    """
    Xy = df.copy()
    # Feature 1: Log price change (Target generation: P_{t+1}/P_t)
    # diff: x[i] - x[i-1]
    Xy["log_close"] = np.log(df["cl"]).diff(1).shift(-1)
    Xy.dropna(inplace=True)
    # Divide the Independent variable from the explanatory ones
    X = Xy.drop(["log_close", "cl"], axis=1)
    y = Xy["log_close"]
    # [!NOTE]: shffule = False to prevent Leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestRegressor(n_estimators = 100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    plot_actual_vs_prediction(y_test, y_pred)
    plot_feature_importances(rf, X_train.columns)
    return y_pred

def plot_feature_importances(model, feature_names):
    """
    Visualizes the contribution of each feature to the model's predictions.
    
    Args:
        model: The trained RandomForestRegressor/Classifier.
        feature_names (list): List of column names (e.g., X_train.columns).
    """
    # 1. Extract and sort importances
    importances = model.feature_importances_
    indices = np.argsort(importances[::-1])
    # 2. Manage results with a key-value pair
    feature_ranking = pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    })
    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(feature_ranking["Feature"], feature_ranking["Importance"], color="skyblue")
    plt.gca().invert_yaxis()  # Put highest importance at the top
    plt.xlabel("Gini Importance / Mean Decrease in Impurity")
    plt.title("Contribution of Features to Random Forest Decisions")
    plt.tight_layout()
    plt.show()
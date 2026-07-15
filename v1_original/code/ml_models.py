from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def train_models(X, y):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    nn = MLPRegressor(
        hidden_layer_sizes=(128,64,32),
        max_iter=1000,
        random_state=42
    )

    rf.fit(X, y)
    nn.fit(X, y)

    return rf, nn
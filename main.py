from data_loading import load_and_prepare_data
from modeling import train_model
from visualization import plot_correlation, plot_predictions


X_train, X_test, y_train, y_test, data = load_and_prepare_data()


model, y_pred, mse, r2 = train_model(X_train, y_train, X_test, y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


plot_correlation(data)
plot_predictions(y_test, y_pred)

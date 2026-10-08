import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


ROLL = "BT2024266"
BASE_DIR = Path(__file__).resolve().parent

TRAIN_VAR1 = BASE_DIR / f"{ROLL}_train_var1.csv"
TEST_VAR1  = BASE_DIR / f"{ROLL}_test_var1.csv"
TRAIN_VAR2 = BASE_DIR / f"{ROLL}_train_var2.csv"
TEST_VAR2  = BASE_DIR / f"{ROLL}_test_var2.csv"


def load_data(train_file, test_file):
    train = np.loadtxt(train_file, delimiter=",", skiprows=1)
    test = np.loadtxt(test_file, delimiter=",", skiprows=1)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test

    return X_train, y_train, X_test


def choose_model(X, y, degree_range, alphas):
    """
    Choose polynomial degree and Ridge regularization using 5-fold CV.
    Ridge is still polynomial regression: we just regularize the
    coefficients of the polynomial features to reduce overfitting.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_mse = float("inf")
    best_degree = None
    best_alpha = None

    for degree in degree_range:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        for alpha in alphas:
            fold_mse = []

            for train_idx, val_idx in kf.split(X_poly):
                model = Ridge(alpha=alpha)
                model.fit(X_poly[train_idx], y[train_idx])

                pred = model.predict(X_poly[val_idx])
                fold_mse.append(mean_squared_error(y[val_idx], pred))

            avg_mse = np.mean(fold_mse)

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_degree = degree
                best_alpha = alpha

    # Train the final model on all training data
    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = Ridge(alpha=best_alpha)
    model.fit(X_poly, y)

    return poly, model, best_degree, best_alpha, best_mse


def train_and_predict(train_file, test_file, degree_range, alphas, output_file):
    X_train, y_train, X_test = load_data(train_file, test_file)

    poly, model, degree, alpha, cv_mse = choose_model(
        X_train, y_train, degree_range, alphas
    )

    # Useful training metrics
    train_pred = model.predict(poly.transform(X_train))
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Predict test data
    test_pred = model.predict(poly.transform(X_test))

    # Submission file: only the predicted y column
    np.savetxt(
        output_file,
        test_pred,
        delimiter=",",
        header="y",
        comments="",
        fmt="%.10f"
    )

    print(f"\n{output_file}")
    print(f"Chosen degree : {degree}")
    print(f"Chosen alpha  : {alpha}")
    print(f"CV MSE        : {cv_mse:.6f}")
    print(f"Train MSE     : {train_mse:.6f}")
    print(f"Train R^2     : {train_r2:.6f}")
    print(f"Saved {len(test_pred)} predictions.")


# Phase 1: 6 input variables, degree up to 10
# The data is much more stable around degree 5.
train_and_predict(
    TRAIN_VAR1,
    TEST_VAR1,
    degree_range=range(1, 11),
    alphas=[1, 3, 10],
    output_file=BASE_DIR / f"{ROLL}_pred_var1.csv"
)

# Phase 2: 3 input variables, degree up to 20
# Higher degrees are needed here, but CV prevents unnecessary overfitting.
train_and_predict(
    TRAIN_VAR2,
    TEST_VAR2,
    degree_range=range(1, 21),
    alphas=[0.03, 0.1, 0.3],
    output_file=BASE_DIR / f"{ROLL}_pred_var2.csv"
)

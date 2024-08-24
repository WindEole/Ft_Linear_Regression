"""Linear regression script.

Ce script entraîne un modèle de régression linéaire pour prédire les prix des
voitures en fonction de leur kilométrage.
Il utilise la descente de gradient pour ajuster les paramètres du modèle.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_squared_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate the loss based on the predictions made.

    In linear regression, we use mean squared error (= sum of the squared
    differences between true and predicted values) to calculate the loss.
    """
    return (np.sum((y_true - y_predicted) ** 2) / len(y_true))


def gradient_descent(
        data: pd.DataFrame,
        learning_rate: float,
        iterations: int,
        stop_threshold: float) -> tuple:
    """Gradient descent.

    Iterative algorithm used to minimize a function by finding the optimal
    parameters. For a linear function (y = a * x + b) -> find a and b.
    iterations, learning_rate, and stop_threshold are the tuning parameters for
    the gradient descent algorithm and can be tuned by the user.
    """
    # Extraction des valeurs de km et price (reduction d'échelle)
    scale = 1e4
    x = data["km"] / scale
    y = data["price"] / scale
    print(x.head())
    print(y.head())

    current_a = 0.1
    current_b = 0.01
    m = float(len(x))
    costs = []
    all_coeff_a = []
    prev_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):

        # Making prediction
        y_predicted = (current_a * x) + current_b

        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)

        # If the change in cost is less than or equal to stop_threshold : stop!
        if prev_cost and abs(prev_cost - current_cost) <= stop_threshold:
            break

        prev_cost = current_cost
        costs.append(current_cost)
        all_coeff_a.append(current_a)

        # Calculate the gradients
        grad_a = - (2 / m) * sum(x * (y - y_predicted))
        grad_b = - (2 / m) * sum(y - y_predicted)

        # Updating coeff a and b
        current_a = current_a - (learning_rate * grad_a)
        current_b = current_b - (learning_rate * grad_b)

        # Printing parameters for every 100th iteration
        if i % 100 == 0:
            print(f"Iteration {i}: tetha0 = {current_b}, tetha1 = {current_a},"
                  f" cost = {current_cost}")

    # Display coeff a and cost
    plt.plot(all_coeff_a, costs)
    plt.scatter(all_coeff_a, costs, marker="o", color="red")
    plt.title("Cost vs coef a (tetha1)")
    plt.ylabel("Cost")
    plt.xlabel("Coeff a")
    plt.show()

    return current_b, current_a  # ATTENTION : tuple inversé car tetha0, tetha1


# def normalize(data: pd.DataFrame) -> pd.DataFrame:
#     """Normalize features to have zero mean and unit variance."""
#     data_norm = data.copy()  # Avoid modifying the original DataFrame
#     print(data_norm.head())

#     # pour la colonne km
#     mean_km = data["km"].mean()
#     std_km = data["km"].std()
#     data_norm["km"] = (data["km"] - mean_km) / std_km

#     # pour la colonne price
#     mean_price = data["price"].mean()
#     std_price = data["price"].std()
#     data_norm["price"] = (data["price"] - mean_price) / std_price

#     print(data_norm.head())
#     return data_norm

def load(path: str) -> pd.DataFrame:
    """Load a file.csv and return a dataset."""
    try:
        data_read = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file {path} is corrupted.")
        return None
    except MemoryError:
        print(f"Error: The file {path} is too large to fit into memory.")
        return None
    except IOError:
        print(f"Error: Unable to open the file at path {path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    data = pd.DataFrame(data_read)

    lines, col = data.shape
    print(f"Loading dataset of dimensions ({lines}, {col})")

    return data


def main() -> None:
    """Load data and send them to gradient algorithm."""
    data = load("data.csv")
    if data is None:
        return
    print(data)

    # Données trop grandes et disparates : on normalise le tout !
    # data_norm = normalize(data)

    # Parametres de la descente de gradient
    learning_rate = 0.00001
    iterations = 2000
    stop_threshold = 0.000001

    # y = b + a * x => b = tetha0 (biais) / a = tetha1 (slope ou weight)
    tetha0, tetha1 = gradient_descent(
        data, learning_rate, iterations, stop_threshold,
        )
    print(f"Trained parameters: tetha0 = {tetha0}, tetha1 = {tetha1}")


    # On enregistre les coeff dans un fichier json avec pathlib
    tetha_values = {
        "tetha0": tetha0,
        "tetha1": tetha1,
    }

    # Definir le chemin du fichier JSON
    file_path = Path("tetha_values.json")

    # Save datas in file.json (s'il n'existe pas, il sera créé automatiquement)
    with file_path.open("w") as file:
        json.dump(tetha_values, file)

    # Making predictions using estimated parameters
    y_prediction = tetha0 + tetha1 * data["km"].to_numpy()

    # Display the regression line
    plt.scatter(
        data["km"],
        data["price"],
        marker="o",
        color="blue",
        label="Data Points",
    )
    plt.plot([min(data["km"].values), max(data["km"].values)],
            [min(y_prediction), max(y_prediction)],
            color="green",markerfacecolor="red",
            markersize=10,linestyle="dashed")
    plt.title("Prices or cars vs. Mileage")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

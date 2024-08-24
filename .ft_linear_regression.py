"""Linear regression script.

Ce script entraîne un modèle de régression linéaire pour prédire les prix des
voitures en fonction de leur kilométrage.
Il utilise la descente de gradient pour ajuster les paramètres du modèle.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Programme qui entraine notre modele de prediction des prix en fonction du
# kilometrage.
# Entrée : des données (tuple) prix / kilométrage (fichier.csv)
# Sortie : tetha0 1, les variables d'ajustement pour la prédiction du prix.

# LINEAR REGRESSION :
# In linear regression, gradient descent is used to find the optimal coef
# that minimize the sum of squared errors between predicted and actual values.
# La régression linéaire est un algorithme qui va trouver une droite qui se
# rapproche le plus possible d'un ensemble de points. Les points représentent
# les données d'entraînement.


# Ceci est la fonction que l'on veut optimiser
def estimate_price(km: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Estimate the price with parameters tetha0 and tetha1."""
    return t0 + t1 * km


def cost_function(
        km: np.ndarray,
        price: np.ndarray,
        t0: float,
        t1: float) -> float:
    """Calculate the cost function."""
    predictions = estimate_price(km, t0, t1)
    # print(f"predictions in cost function = {predictions}")
    error = predictions - price
    # print(f"error in cost function = {error}")
    return np.mean(error ** 2) / 2


def gradient_descent(data: pd.DataFrame, learning_rate: float, iterations: int) -> tuple:
    """Gradient descent."""

    # Extraction des valeurs de km et price
    km = data['km'].values
    print(f"km = {km}")
    price = data['price'].values
    print(f"price = {price}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(km, price)
    print(f"slope = {slope}, intercept = {intercept}")

    # Verifier si valeurs manquantes ou invalides
    if np.any(np.isnan(km)) or np.any(np.isnan(price)):
        raise ValueError("Data contains NaN values.")
    if np.any(np.isinf(km)) or np.any(np.isinf(price)):
        raise ValueError("Data contains infinite values.")

    # Initialisation des paramètres
    m = len(km)
    tetha0 = 0
    tetha1 = 0
    cost_history = []  # Liste pour stocker les valeurs de la fonction de cout

    # Affichage de la droite de regression via matplotlib
    # plt.ion()  # active le mode interactif de Matplotlib
    # fig, ax = plt.subplots()
    # ax.scatter(km, price, marker="o", color="blue", label="Data Points")
    # plt.title("Prices of Cars vs. km")
    # plt.xlabel("km")
    # plt.ylabel("Price")
    # line, = ax.plot([], [], color="red", label="Regression Line")  # Ligne initiale vide
    # plt.legend()

    ###########################################################################
    # Formules à implémenter pour descente de gradient :
    # tmpθ0 = learningRate * 1/m m-1∑i=0(estimPrice(km[i]) - price[i])
    # tmpθ1 = learningRate * 1/m m-1∑i=0(estimPrice(km[i]) - price[i]) * km[i]
    ###########################################################################

    for i in range(iterations):
        # predictions actuelles
        predictions = estimate_price(km, tetha0, tetha1)

        # calcul des erreurs
        error = predictions - price

        # calcul des gradients
        grad0 = (learning_rate / m) * np.sum(error)
        grad1 = (learning_rate / m) * np.sum(error * km)

        # mise à jour des paramètres
        tetha0 -= grad0
        tetha1 -= grad1

        # Enregistrer la fonction de coût
        cost = cost_function(km, price, tetha0, tetha1)
        cost_history.append(cost)

        # Affichage de la progression
        if i % 100 == 0:
            print(f"Iteration {i}: tetha0 = {tetha0}, tetha1 = {tetha1}, cost = {cost}")

            # Màj de la ligne de regression
            # x_values = np.linspace(km.min(), km.max(), 100)
            # y_values = tetha0 + tetha1 * x_values
            # line.set_xdata(x_values)
            # line.set_ydata(y_values)
            # fig.canvas.draw()  # Redessine la figure
            # fig.canvas.flush_events()  # Assure l'affichage des elements

        # Vérifiez si les valeurs des gradients contiennent infini ou NaN
        if np.isnan(tetha0) or np.isnan(tetha1) or np.isinf(tetha0) or np.isinf(tetha1):
            raise ValueError("Calculated theta values contain NaN or infinite values.")

    # plt.ioff()  # Desactive le mode interactif de Matplotlib
    # plt.show()  # Affiche le graphe final
    return tetha0, tetha1, cost_history


def denormalize_coeff(tetha0: float, tetha1: float, mean:float, std: float) -> tuple:
    """Adjust the tetha1 coeff to match the original data scale."""
    tetha1_denorm = tetha1 / std
    tetha0_denorm = tetha0 - (tetha1 * mean / std)
    return tetha0_denorm, tetha1_denorm


def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to have zero mean and unit variance."""
    data = data.copy()  # Avoid modifying the original DataFrame
    for column in ["km"]:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = (data[column] - mean) / std
    # mean = data["km"].mean()
    # std = data["km"].std
    # data["km_norm"] = (data["km"] - mean) / std
    return data, mean, std


def display(data: pd.DataFrame, tetha0: float, tetha1: float) -> None:
    """Display data in a graph."""
    try:
        if "km" not in data.columns or "price" not in data.columns:
            raise ValueError("DataFrame must contain 'km' and 'price' column.")

        plt.scatter(
            data["km"],
            data["price"],
            marker="o",
            color="blue",
            label="Data Points",
            )
        # plt.plot(
        #     data["km"],
        #     estimate_price(data["km"].values, tetha0, tetha1),
        #     color="red",
        #     label="Regression Line",
        #     )
        plt.title("Prices or cars vs. Mileage")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        plt.close()
    except Exception as e:
        print(f"An unexpected error occurred : {e}")


def load(path: str) -> pd.DataFrame:
    """Load and return a dataset."""
    try:
        data = pd.read_csv(path)
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

    dataFrame = pd.DataFrame(data)

    lines, col = dataFrame.shape
    print(f"Loading dataset of dimensions ({lines}, {col})")

    return dataFrame


def main():
    dataFrame = load("data.csv")
    if dataFrame is None:
        return

    print(dataFrame)


    # Normalisation des données
    dataFrame, mean, std = normalize_features(dataFrame)

    # Parametres de la descente de gradient
    learning_rate = 0.000001
    iterations = 1000
    tetha0, tetha1, cost_history = gradient_descent(
        dataFrame, learning_rate, iterations,
        )

    # il faut denormaliser le coefficient tetha1 !
    tetha0, tetha1 = denormalize_coeff(tetha0, tetha1, mean, std)

    print(f"Trained parameters: tetha0 = {tetha0}, tetha1 = {tetha1}")
    tetha_values = {
        "tetha0": tetha0,
        "tetha1": tetha1,
    }
    with open("tetha_values.json", "w") as file:
        json.dump(tetha_values, file)
    display(dataFrame, tetha0, tetha1)


if __name__ == "__main__":
    main()

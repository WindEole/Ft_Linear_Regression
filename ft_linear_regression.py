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


def close_on_enter(event: any) -> None:
    """Close the figure when the Enter key is pressed."""
    if event.key == "enter":  # Si la touche 'Enter' est pressée
        plt.close(event.canvas.figure)  # Ferme la figure associée


def mean_squared_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate the loss based on the predictions made.

    In linear regression, we use mean squared error (= sum of the squared
    differences between true and predicted values) to calculate the loss.
    """
    # return (np.sum((y_true - y_predicted) ** 2) / len(y_true))
    mse = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    # On ajoute une pénalité en cas de prediction négative !
    negative_predictions = y_predicted[y_predicted < 0]
    # la pénalité sera proportionnelle au carré des valeurs prédites négatives
    penalty_factor = 10.0
    penalty = penalty_factor * np.sum(negative_predictions ** 2)
    return mse + penalty


def mean_absolute_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate the mean absolute error (MAE) based on the predictions made.

    In linear regression, we use mean absolute error (average of the absolute
    differences between true and predicted values) to calculate the error.
    """
    return np.sum(np.abs(y_true - y_predicted)) / len(y_true)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate the performance of the linear regression model."""
    print("\033[94m\nEVALUATION DU MODELE :\033[0m")

    # Mean Squared Error
    print("\033[96m\nPremier indicateur : MSE = mean square error :\033[0m")
    print("MSE évalue la différence entre valeurs réelles et prédites.")
    mse = mean_squared_error(y_true, y_pred)
    print(f"\033[91mMSE: {mse}\033[0m. Plus cet indicateur est petit, mieux "
          "c'est. Ici, il est très grand, car il y a des valeurs aberrantes "
          "dans le dataset.")

    # Mean Absolute Error
    print("\033[96m\nDeuxième indicateur : MAE = mean absolute error :\033[0m")
    print("MAE évalue l'erreur moyenne entre valeurs prédites et réelles.")
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE: {mae} ce qui signifie qu'il peut y avoir une différence ")
    print(f"de \033[91m{int(mae)}\033[0m euros entre prix réels et prédits.")

    # R-squared score
    print("\033[96m\nTroisième indicateur : R-squared Score (R²) :\033[0m"
          "R² évalue combien de variance des data est capturée par le modèle."
          "\nR² ~ 1 -> le modèle s'ajuste bien aux données,"
          "\nR² ~ 0 -> le modèle n'explique presque rien.")
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    print(f"La précision de notre modèle est : R-squared (R²): {r2}")
    r2_round = round(r2, 4)
    print(
        f"donc \033[91m{r2_round * 100}%\033[0m des variations de prix est "
        "prise en compte.",
    )

    # print("\033[96m\nConclusion :\033[0m")
    # print("Ces trois indicateurs montrent que le modèle est relativement "
    #       "efficace dans le cadre d'une régression linéaire à partir de "
    #       "données assez éparses. Le modèle pourrait être affiné soit en "
    #       "ajoutant d'autres critères comme l'âge du véhicule ou la marque, "
    #       "soit en passant à un autre modèle de régression (non linéaire)")
    return r2


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
    # Normaliser les données car ordres de grandeurs trop différents :
    # km (max 250000) et prix (max 10000) déstabilise la descente de gradient.
    km_mean = data["km"].mean()  # Moyenne = sum of all entries / nb of entries
    km_std = data["km"].std()  # Ecart Type = sqroot(mean(abs(x -x.mean())^2))
    price_mean = data["price"].mean()
    price_std = data["price"].std()
    data["km_normalized"] = (data["km"] - km_mean) / km_std
    data["price_normalized"] = (data["price"] - price_mean) / price_std

    x = data["km_normalized"]
    y = data["price_normalized"]

    # Init qui fonctionne : current_a = -1 | current_b = 0.1 ATT BIAIS
    current_a = np.random.randn() * 0.01
    current_b = np.random.randn() * 0.01
    m = float(len(x))
    costs = []
    all_coeff_a = []
    prev_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):
        y_predicted = (current_a * x) + current_b  # Making prediction
        current_cost = mean_squared_error(y, y_predicted)  # Calcul curent cost

        # If the change in cost is less than or equal to stop_threshold : stop!
        if prev_cost and abs(prev_cost - current_cost) <= stop_threshold:
            break

        prev_cost = current_cost
        costs.append(current_cost)
        all_coeff_a.append(current_a)

        # Calcul des gradients : un gradient indique dans quelle direction
        # (et avec quelle intensité) il faut modifier les coefficients pour
        # diminuer l'erreur (dérivées partielles de la fonction de coût).
        grad_a = - (2 / m) * sum(x * (y - y_predicted))
        grad_b = - (2 / m) * sum(y - y_predicted)

        # Updating coeff a and b
        current_a = current_a - (learning_rate * grad_a)
        current_b = current_b - (learning_rate * grad_b)

        # Printing parameters for every 100th iteration
        if i % 100 == 0:
            # print(f"Iteration {i}: t0 = {current_b}, t1 = {current_a},"
            #       f" cost = {current_cost}")
            plt.cla()  # Efface le graphe précédent
            plt.scatter(x, y, color="blue", label="True Data")  # Nuage de pts
            plt.plot(  # Ici on a la droite de régression linéaire
                x,
                y_predicted,
                color="green",
                label="Prediction Line",
            )
            plt.title(f"Iteration {i}: Regression Line Progression")
            plt.xlabel("Mileage Normalized")
            plt.ylabel("Price Normalized")
            plt.legend()
            plt.pause(0.001)  # Forcer l'affichage du graphique actualisé

    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show()  # Final Display for the last iteration

    # Conversion des coefficients après normalisation
    final_a = current_a * (price_std / km_std)
    final_b = (current_b * price_std) + price_mean - (final_a * km_mean)
    # print(f"final_a = {final_a}, final_b = {final_b}")

    # Display coeff a and cost
    plt.figure()
    plt.plot(all_coeff_a, costs)
    # plt.scatter(all_coeff_a, costs, marker="|", color="red")
    plt.title("Cost vs coef a (tetha1)")
    plt.ylabel("Cost")
    plt.xlabel("Coeff a")
    # Close with Enter
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show(block=False)
    plt.pause(0.001)

    return final_b, final_a  # ATTENTION : tuple inversé car tetha0, tetha1


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

    # Parametres de la descente de gradient
    learning_rate = 0.0001
    iterations = 30000  # donne une précision de 73.26% | 30000 -> 73.3%
    stop_threshold = 0.000001

    # y = b + a * x => b = tetha0 (biais) / a = tetha1 (slope ou weight)
    tetha0, tetha1 = gradient_descent(
        data, learning_rate, iterations, stop_threshold,
        )
    print(
        f"\033[91mTrained parameters: tetha0 = {tetha0}, "
        f"tetha1 = {tetha1}\033[0m",
    )

    # On crée un tuple pour les 2 coeff
    tetha_values = {
        "tetha0": tetha0,
        "tetha1": tetha1,
    }

    # Save datas in file.json avec pathlib (s'il n'existe pas, il sera créé)
    file_path = Path("tetha_values.json")
    with file_path.open("w") as file:
        json.dump(tetha_values, file)

    # Making predictions using estimated parameters
    y_prediction = tetha0 + tetha1 * data["km"].to_numpy()

    # BONUS PART : Évaluer la précision du modèle
    evaluate_model(data["price"].to_numpy(), y_prediction)

    # Final Display : regression line vs true Data
    plt.figure()
    plt.scatter(  # Ici on affiche les données d'origine
        data["km"],
        data["price"],
        marker="o",
        color="blue",
        label="Original Data",
    )
    plt.scatter(  # Ici on affiche les données après prédiction
        data["km"],
        y_prediction,
        marker="|",
        color="red",
        label="Predicted Data",
    )
    plt.plot(  # Ici on a la droite de régression linéaire
        data["km"],
        y_prediction,
        color="green",
        label="Prediction Line",
    )
    plt.title("Mileage vs Prices")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    fig = plt.gcf()  # On obtient le graphe en cours
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessus interrompu par l'utilisateur.")

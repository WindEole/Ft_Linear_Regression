"""prediction script.

Ce programme prédit le prix d'une voiture en fonction de son kilométrage
Entrée : kilométrage.
Sortie : le prix.
"""

import json


class Tetha:
    tetha0 = 0
    tetha1 = 0


def load_tetha_values():
    """Load tetha values from a JSON file."""
    try:
        with open("tetha_values.json", "r") as file:
            tetha_values = json.load(file)
            Tetha.tetha0 = tetha_values["tetha0"]
            Tetha.tetha1 = tetha_values["tetha1"]
    except FileNotFoundError:
        print("Error: Required file (tetha_values.json) is missing")
        raise


def price_prediction(mileage: int) -> int:
    """Predict the price of a car with a given mileage."""
    price = Tetha.tetha0 + (Tetha.tetha1 * mileage)
    return int(price)


def main():

    try:
        # 1) charger les valeurs de tetha
        load_tetha_values()

        # 2) Recupérer l'info utilisateur
        mileage = input("Enter the mileage of your car: ")

        # 3) Validation de l'entrée
        try:
            mileage = int(mileage)
        except ValueError:
            print("Invalid mileage input. Please enter an integer number.")
            return

        print(f"The mileage you gave is {mileage}")

        # 3) Calcul du prix -> function
        price = price_prediction(mileage)
        print(f"A car with {mileage} miles is worth {price} dollars.")

    except FileNotFoundError:
        print("Critical error: Please train your model.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by the user.")


if __name__ == "__main__":
    main()

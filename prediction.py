import json

# Programme qui prédit le prix d'une voiture en fonction de son kilométrage.
# Entrée : kilométrage
# Sortie : le prix.


class Tetha:
    tetha0 = 0
    tetha1 = 0


def load_tetha_values():
    """Load tetha values from a JSON file."""
    with open("tetha_values.json", "r") as file:
        tetha_values = json.load(file)
        Tetha.tetha0 = tetha_values["tetha0"]
        Tetha.tetha1 = tetha_values["tetha1"]


def price_prediction(mileage: int) -> int:
    """Predict the price of a car with a given mileage."""
    # estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
    price = Tetha.tetha0 + (Tetha.tetha1 * mileage)
    return price


def main():

    # 1) charger les valeurs de tetha
    load_tetha_values()

    # 2) Recupérer l'info utilisateur
    mileage = input("Enter the mileage of your car: ")

    # 3) Validation de l'entrée
    try:
        mileage = int(mileage)
    except ValueError:
        print("Invalid mileage input. Please enter a number.")
        return

    # Affichage
    print(f"The mileage you gave is {mileage}")

    # 3) Calcul du prix -> function
    price = price_prediction(mileage)
    print(f"A car with {mileage} miles is worth {price} dollars.")

if __name__ == "__main__":
    main()
import pandas as pd
import os
import datetime

def main():
    print("Inicio de ejecucion.")

    # Cargar datos
    try:
        df = pd.read_csv("data/competencia_01.csv")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")


    # Mostrar datos
    print(df.head())
    print(f"Filas:{df.shape[0]} Columnas:{df.shape[1]}")
    print("Dataset cargado.")

    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} - Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas\n")
    
    print(">>> Ejecución finalizada. Revisa logs/logs.txt")

if __name__ == "__main__":
    main()
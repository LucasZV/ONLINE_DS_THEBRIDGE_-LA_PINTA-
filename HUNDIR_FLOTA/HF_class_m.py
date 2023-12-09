import numpy as np

# DEFINIMOS LA CLASE BARCO

class Barco:
    def __init__(self,nombre,longitud,coordenadas=[]):
        self.nombre=nombre
        self.longitud=longitud
        self.vidas=[False]*longitud
        self.coord=coordenadas

    def tocado(self,posicion):
        if 0 <= posicion < self.longitud:
            self.celdas[posicion] = True
            print(f"¡El barco {self.nombre} fue golpeado en la posición {posicion}!")
        else:
            print("Posición fuera de rango. No se realizó el golpe.")

    def esta_hundido(self):
        return all(self.vidas)        

#DEFINIMOS LA CLASE TABLERO 
class Tablero:
    def __init__(self,nombre,filas, columnas):
        self.nombre=nombre
        self.filas = filas
        self.columnas = columnas
        self.matriz = [['O' for _ in range(columnas)] for _ in range(filas)]

    def crea_tablero(pieza=" "):
        dimensiones=TAM_TABLERO=(10,10)
        return np.full(dimensiones,pieza)

    def colocar_barco(self, barco, fila, columna, orientacion): #colocar barco
        
        if orientacion == 'horizontal':
            for i in range(barco.longitud):
                self.matriz[fila][columna + i] = barco.nombre[0]
        elif orientacion == 'vertical':
            for i in range(barco.longitud):
                self.matriz[fila + i][columna] = barco.nombre[0]
        else:
            print("Orientación no válida. Use 'horizontal' o 'vertical'.")

    def recibir_disparo(self, fila, columna):
        if 0 <= fila < self.filas and 0 <= columna < self.columnas:
            if self.matriz[fila][columna] == 'O':
                print("¡Agua!")
                return False
            elif self.matriz[fila][columna] == 'X':
                print("Ya has disparado a esta posición.")
                return False
            else:
                print(f"¡Impacto en {fila}, {columna}!")
                self.matriz[fila][columna] = 'X'
                return True
        else:
            print("Posición fuera de rango. Disparo no válido.")
            return False

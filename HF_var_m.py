
import HF_class_m as clases
import numpy as np



#LISTADO DE VARIABLES

""" 1. Necesitarás un conjunto de **constantes**, donde tengas inventariados los barcos del juego, dimensiones y demás variables que no vayan a cambiar que tendréis definidas en archivo de **variables.py** """

#DIMENSIONES DEL TABLERO

filas=10
columnas=10
tamaño_tablero=(filas,columnas)
tablero = np.full((filas,columnas), ' ')
tablero_maquina = np.full((filas,columnas),' ')
tablero_visible = np.full((filas,columnas), '?')


#BARCOS

barco4_1 = clases.Barco("Portaviones",4)
barco3_1 = clases.Barco("Crucero",3)
barco3_2 = clases.Barco("Acorazado",3)
barco2_1 = clases.Barco("Destructor",2)
barco2_2 = clases.Barco("Cañonero",2)
barco2_3 = clases.Barco("Submarino",2)
barco1_1 = clases.Barco("Fragata",1)
barco1_2 = clases.Barco("Lancha",1)
barco1_3 = clases.Barco("Nodriza",1)
barco1_4 = clases.Barco("Anfibio",1)

barcos_lista=[barco4_1,barco3_1,barco3_2,barco2_1,barco2_2,barco2_3,barco1_1,barco1_2,barco1_3,barco1_4]
tamaño_barcos=[4,3,3,2,2,2,1,1,1,1]
diccionario_barcos={"Portaviones":barco4_1,
                    "Crucero":barco3_1,
                    "Acorazado":barco3_2,
                    "Destructor":barco2_1,
                    "Cañonero":barco2_2,
                    "Submarino":barco2_3,
                    "Fragata":barco1_1,
                    "Lancha":barco1_2,
                    "Nodriza":barco1_3,
                    "Anfibio":barco1_4}


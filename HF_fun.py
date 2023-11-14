import numpy as np
import time

#CREACION DEL TABLERO

tablero = np.full((10,10), ' ')
tablero_maquina = np.full((10,10),' ')
tablero_visible = np.full((10,10), '?')

TAM_TABLERO=(10,10)

def crea_tablero(pieza=" "):
    dimensiones=TAM_TABLERO=(10,10)
    return print(np.full(dimensiones,pieza))

def crea_tablero_maquina(tile=" "):
    dimensiones=TAM_TABLERO=(10,10)
    return print(np.full(dimensiones,tile))

def limpiar_tablero(quetablero,pieza=" "):
        quetablero=np.full(TAM_TABLERO,pieza)
        return print(quetablero)

def coloca_barco(tablero,barco):
    for pieza in barco:
        tablero[pieza]="O"

def generar_barco_simple (tablero,eslora,tablero_auto=True):

    #Decido si le quiero dar orientacion aleatoria o no
    if tablero_auto:
        orientacion = np.random.choice(['N','S','E','O'])
    elif tablero_auto==False:
        orientacion = input('Introduce una de estas direcciones: N, S, E, O')
    else:
        print("Valor erroneo: Introduce una de estas direcciones: N, S, E, O")
        orientacion=input()

    #para cada orientacion (auto o manual) le doy unas cordenadas al barco y lo meto en el tablero
    #si no tengo espacio para mas barcos se queda pensando que hacer, tengo que corregir eso
    if orientacion == 'E':
        while True:
            if tablero_auto:
                barco_x = np.random.randint(0,10)
                barco_y = np.random.randint(0,10)
            else:
                barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))

            if barco_y <= (10-eslora) and all(hueco != 'O' for hueco in tablero[barco_x , barco_y : barco_y + eslora]):
                tablero[barco_x , barco_y : barco_y + eslora] = 'O'
                break
    
    elif orientacion == 'O':
        while True:
            if tablero_auto:
                barco_x = np.random.randint(0,10)
                barco_y = np.random.randint(0,10)
            else:
                barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
            if barco_y >= (eslora-1) and all(hueco != 'O' for hueco in tablero[barco_x, barco_y - eslora : barco_y]):
                tablero[barco_x, barco_y - eslora : barco_y] = 'O'
                break

    elif orientacion == 'N':
        while True:
            if tablero_auto:
                barco_x = np.random.randint(0,10)
                barco_y = np.random.randint(0,10)
            else:
                barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
            if barco_x >= (eslora-1) and all(hueco != 'O' for hueco in tablero[barco_x - eslora : barco_x, barco_y]):
                tablero[barco_x - eslora : barco_x, barco_y] = 'O'
                break
        
    elif orientacion == 'S':
        while True:
            if tablero_auto:
                barco_x = np.random.randint(0,10)
                barco_y = np.random.randint(0,10)
            else:
                barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
            if barco_x <= (10-eslora) and all(hueco != 'O' for hueco in tablero[barco_x : barco_x + eslora, barco_y]):
                tablero[barco_x : barco_x + eslora, barco_y] = 'O'
                break

#imprimo el tablero al final, los barcos se me solapan
def generar_todos_los_barcos(tablero_auto=True):
    cantidad_barcos_4_eslora = 1
    cantidad_barcos_3_eslora = 2
    cantidad_barcos_2_eslora = 3
    cantidad_barcos_1_eslora = 4

    if tablero_auto==False:
        for eslora in range(cantidad_barcos_4_eslora):
            print('Introduce las coordenadas de tu barco de 4 de eslora')
            generar_barco_simple(tablero,4,tablero_auto=False)  #lo tengo que generar en mi tablero
            #print(tablero)   #mi tablero tambien
            #print("*********************************************")

        for eslora in range(cantidad_barcos_3_eslora):
            print('Introduce las coordenadas de tu barco de 3 de eslora')
            generar_barco_simple(tablero,3,tablero_auto=False)
            #print(tablero)
            #print("*********************************************")

        for eslora in range(cantidad_barcos_2_eslora):
            print('Introduce las coordenadas de tu barco de 2 de eslora')
            generar_barco_simple(tablero,2,tablero_auto=False)
            #print(tablero)
            #print("*********************************************")

        for eslora in range(cantidad_barcos_1_eslora):
            print('Introduce las coordenadas de tu barco de 1 de eslora')
            generar_barco_simple(tablero,1,tablero_auto=False)
            #print(tablero)
            #print("*********************************************")

    elif tablero_auto==True:
        for eslora in range(cantidad_barcos_4_eslora):
            generar_barco_simple(tablero,4,tablero_auto=True)  #lo tengo que generar en mi tablero
            #print(tablero)   #mi tablero tambien
            #print("*********************************************")

        for eslora in range(cantidad_barcos_3_eslora):
            generar_barco_simple(tablero,3,tablero_auto=True)
            #print(tablero)
            #print("*********************************************")

        for eslora in range(cantidad_barcos_2_eslora):
            generar_barco_simple(tablero,2,tablero_auto=True)
            #print(tablero)
            #print("*********************************************")

        for eslora in range(cantidad_barcos_1_eslora):
            generar_barco_simple(tablero,1,tablero_auto=True)
            #print(tablero)
            #print("*********************************************")
    print(tablero)
    print("*********************************************")
            
# GENERA TODOS LOS BARCOS DE LA MAQUINA
def generar_todos_los_barcos_maquina():
    cantidad_barcos_4_eslora = 1
    cantidad_barcos_3_eslora = 2
    cantidad_barcos_2_eslora = 3
    cantidad_barcos_1_eslora = 4
    
    for _ in range(cantidad_barcos_4_eslora):
        generar_barco_simple(tablero_maquina,4,tablero_auto=True)
    for _ in range(cantidad_barcos_3_eslora):
        generar_barco_simple(tablero_maquina,3,tablero_auto=True)
    for _ in range(cantidad_barcos_2_eslora):
        generar_barco_simple(tablero_maquina,2,tablero_auto=True)
    for _ in range(cantidad_barcos_1_eslora):
        generar_barco_simple(tablero_maquina,1,tablero_auto=True)

# Tengo que revisarlo, me da fallos
def dispara_propio(tablero_maquina,tablero_visible):
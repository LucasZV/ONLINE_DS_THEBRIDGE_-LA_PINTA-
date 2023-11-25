import numpy as np
import time
import HF_var_m as var

#CREACION DEL TABLERO

tablero = var.tablero
tablero_maquina =var.tablero_maquina
tablero_visible =var.tablero_visible

def crea_tablero(pieza=" "):
    dimensiones=var.tamaño_tablero=(10,10)
    return np.full(dimensiones,pieza)

def crea_tablero_maquina(tile=" "):
    dimensiones=TAM_TABLERO=(10,10)
    return np.full(dimensiones,tile)  

def coloca_barco(tablero,barco):
    for pieza in barco:
        tablero[pieza]="O"


def generar_barco_simple(tablero, eslora, tablero_auto=True):
    if tablero_auto:
        orientacion = np.random.choice(['N', 'S', 'E', 'O'])
    else:
        orientacion = input('Introduce una de estas direcciones: N, S, E, O')

    intentos = 0
    max_intentos = 100  # Número máximo de intentos para encontrar una posición válida

    while intentos < max_intentos:
        if orientacion == 'E':
            if tablero_auto:
                barco_x = np.random.randint(0, tablero.shape[0])
                barco_y = np.random.randint(0, tablero.shape[1] - eslora + 1)
            else:
                while True:
                    barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                    if barco_x > tablero.shape[1]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
                while True:        
                    barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
                    if barco_y + eslora > tablero.shape[0]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
            if all(tablero[barco_x, y] != 'O' for y in range(barco_y, barco_y + eslora)):
                    tablero[barco_x, barco_y:barco_y + eslora] = 'O'
                    return tablero

        elif orientacion == 'O':
            if tablero_auto:
                barco_x = np.random.randint(0, tablero.shape[0])
                barco_y = np.random.randint(eslora - 1, tablero.shape[1])
            else:
                while True:
                    barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                    if barco_x > tablero.shape[1]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
                while True:        
                    barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
                    if barco_y - eslora + 1 < 0:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
            if all(tablero[barco_x, y] != 'O' for y in range(barco_y - eslora + 1, barco_y + 1)):
                tablero[barco_x, barco_y - eslora + 1:barco_y + 1] = 'O'
                return tablero
                
        elif orientacion == 'N':
            if tablero_auto:
                barco_x = np.random.randint(eslora - 1, tablero.shape[0])
                barco_y = np.random.randint(0, tablero.shape[1])
            else:
                while True:
                    barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                    if barco_x - eslora + 1 > 0:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
                while True:        
                    barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
                    if barco_y > tablero.shape[0]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
            if all(tablero[x, barco_y] != 'O' for x in range(barco_x - eslora + 1, barco_x + 1)):
                tablero[barco_x - eslora + 1:barco_x + 1, barco_y] = 'O'
                return tablero

        elif orientacion == 'S':
            if tablero_auto:
                barco_x = np.random.randint(0, tablero.shape[0] - eslora + 1)
                barco_y = np.random.randint(0, tablero.shape[1])
            else:
                while True:
                    barco_x = int(input('Introduce una coordenada X entre 0 y 9 inclusive'))
                    if barco_x + eslora > tablero.shape[1]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break
                while True:        
                    barco_y = int(input('Introduce una coordenada Y entre 0 y 9 inclusive'))
                    if barco_y > tablero.shape[0]:
                        print("Valor no valido, introduce un numero valido")
                    else:
                        break   
            if all(tablero[x, barco_y] != 'O' for x in range(barco_x, barco_x + eslora)):
                tablero[barco_x:barco_x + eslora, barco_y] = 'O'
                return tablero

        intentos += 1

    if intentos >= max_intentos:
        print("No se encontró una posición válida para el barco después de varios intentos.")
    else:
        return tablero
    
#Nueva funcion para generar los barcos
def generar_todos_los_barcos3(tablero_auto=True):
    bl = var.barcos_lista
    coord_globales = []
    for barco in bl:
        generar_barco_simple(tablero, barco.longitud, tablero_auto=tablero_auto)

        barco.coord = []
        for x in range(tablero.shape[0]):
            for y in range(tablero.shape[1]):
                if tablero[x, y] == 'O':
                    coord = (x, y)
                    if (coord not in barco.coord) and (coord not in coord_globales):
                        barco.coord.append(coord)
                        coord_globales.append(coord)
                   
        # Imprimir coordenadas del barco
        print(f"Coordenadas de {barco.nombre}: {barco.coord}")
        print(barco.nombre)
        print(tablero)
        print("*" * 20)

def generar_todos_los_barcos2(tablero_auto=True):
    bl=var.barcos_lista
    tb=var.tamaño_barcos
    if tablero_auto==False:
        for barco in bl:
            generar_barco_simple(tablero,barco.longitud,tablero_auto=False)
    
            coord_barcos = []
            barco_actual = []
            for x in range(tablero.shape[0]):
                for y in range(tablero.shape[1]):
                    if tablero[x, y] == 'O':
                        barco_actual.append((x, y))
                    elif barco_actual:
                        coord_barcos.append(barco_actual)
                        barco_actual = []
            if barco_actual:
                coord_barcos.append(barco_actual)
            for barco, ship_coordinates in zip(bl, coord_barcos):
                print(f"Coordenadas de {barco.nombre}: {ship_coordinates}")

            print(barco.nombre)
            print(tablero)
            print("*"*20)

    elif tablero_auto==True:
        for barco in bl:
            generar_barco_simple(tablero,barco.longitud,tablero_auto=True)
    
            coord_barcos = []
            barco_actual = []
            for x in range(tablero.shape[0]):
                for y in range(tablero.shape[1]):
                    if tablero[x, y] == 'O':
                        barco_actual.append((x, y))
                    elif barco_actual:
                        coord_barcos.append(barco_actual)
                        barco_actual = []
            if barco_actual:
                coord_barcos.append(barco_actual)
            #for i, ship_coordinates in enumerate(coord_barcos, start=1):
                #print(f"Coordenadas de {i}: {ship_coordinates}")
            for barco, ship_coordinates in zip(bl, coord_barcos):
                print(f"Coordenadas de {barco.nombre}: {ship_coordinates}")

            print(barco.nombre)
            print(tablero)
            print("*"*20)


#No se si imprimir 1 por 1 cada barco o todos
def generar_todos_los_barcos(tablero_auto=True):
    bl=var.barcos_lista
    tb=var.tamaño_barcos
    #cantidad_barcos_4_eslora = 1
    #cantidad_barcos_3_eslora = 2
    #cantidad_barcos_2_eslora = 3
    #cantidad_barcos_1_eslora = 4

    if tablero_auto==False:
        for eslora in range(tb.count(4)):
            print('Introduce las coordenadas de tu barco de 4 de eslora')
            generar_barco_simple(tablero,bl[0].longitud,tablero_auto=False)
            print(tablero)
            print("*"*20)

        for eslora in range(tb.count(3)):
            print('Introduce las coordenadas de tu barco de 3 de eslora')
            generar_barco_simple(tablero,bl[1].longitud,tablero_auto=False)
            print(tablero)
            print("*"*20)

        for eslora in range(tb.count(2)):
            print('Introduce las coordenadas de tu barco de 2 de eslora')
            generar_barco_simple(tablero,bl[3].longitud,tablero_auto=False)
            print(tablero)
            print("*"*20)

        for eslora in range(tb.count(1)):
            print('Introduce las coordenadas de tu barco de 1 de eslora')
            generar_barco_simple(tablero,bl[6].longitud,tablero_auto=False)
            print(tablero)
            print("*"*20)

    elif tablero_auto==True:
        for eslora in range(tb.count(4)):
            generar_barco_simple(tablero,bl[0].longitud,tablero_auto=True)  #lo tengo que generar en mi tablero
            print(tablero)   #mi tablero tambien
            print("*"*20)

        for eslora in range(tb.count(3)):
            generar_barco_simple(tablero,bl[1].longitud,tablero_auto=True)
            print(tablero)
            print("*"*20)

        for eslora in range(tb.count(2)):
            generar_barco_simple(tablero,bl[3].longitud,tablero_auto=True)
            print(tablero)
            print("*"*20)

        for eslora in range(tb.count(1)):
            generar_barco_simple(tablero,bl[6].longitud,tablero_auto=True)
            print(tablero)
            print("*"*20)

            
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


def limpiar_tablero(lado=int(10),pieza=" "):
        tablero=np.full((lado,lado),pieza)
        return tablero


# Tengo que revisarlo, me da fallos
def dispara_propio(tablero_maquina,tablero_visible):
    print('Si aciertas te tocará otra vez. Si fallas jugará la máquina')
    turno_jugador = True
    contador_turnos_jugador = 0
    contador_turnos_maquina = 0

    juego_en_progreso=True
    while juego_en_progreso:
        if turno_jugador == True:
            contador_turnos_jugador = contador_turnos_jugador +1
            print(f'el turno actual del jugador es: {contador_turnos_jugador}')
            if 'O' in tablero_maquina:
                while juego_en_progreso:

                    x_1=input('Introduce la coordenada X de tu disparo: [0,9],si deseas salir escribe <salir>')
                    if x_1.lower()=="salir":
                        print("HAS SALIDO DEL JUEGO")
                        limpiar_tablero()
                        juego_en_progreso=False #termino la partida
                        break

                    y_1= input('Introduce la coordenada Y de tu disparo: [0,9], si deseas salir escribe <salir>')
                    if y_1.lower()=="salir":
                        print("HAS SALIDO DEL JUEGO")
                        limpiar_tablero()
                        juego_en_progreso=False #termino la partida
                        break

                    try:
                        disparo_mio_x = int(x_1)
                        disparo_mio_y = int(y_1)
                    except ValueError:
                        print("Se espera un número o salir")
                        continue
                    if 0 <= disparo_mio_x < 10 and 0 <= disparo_mio_y < 10:    
                        if tablero_maquina[disparo_mio_x, disparo_mio_y] == "O":
                            tablero_maquina[disparo_mio_x, disparo_mio_y] = "X"
                            tablero_visible[disparo_mio_x, disparo_mio_y] = 'X'         #no me aparecen mensajes de tocado
                            print('¡TOCADO!\n')
                            for barco in var.barcos_lista:
                                if all(coordinate == "X" for coordinate in barco.coord):
                                    print(f"{barco.nombre} HUNDIDO!\n")

                            print('TABLERO')
                            print(tablero_visible)
                            print('\n')
                            time.sleep(1)
                            
                            turno_jugador = True
                        
                        elif tablero_maquina[disparo_mio_x, disparo_mio_y] == "X":
                            print("Disparo previamente realizado por el jugador!")
                            print('TABLERO')
                            print(tablero_visible)
                            print('\n')
                            time.sleep(1)

                            turno_jugador = True

                        elif tablero_maquina[disparo_mio_x, disparo_mio_y] == "-":
                            print("Disparo previamente realizado por el jugador!")
                            print('TABLERO')
                            print(tablero_visible)
                            print('\n')
                            time.sleep(1)
                            
                            turno_jugador = True

                        elif tablero_maquina[disparo_mio_x, disparo_mio_y] == " ":
                            tablero_maquina[disparo_mio_x, disparo_mio_y] = "-"
                            tablero_visible[disparo_mio_x, disparo_mio_y] = '-'
                            print("¡AGUA!")
                            print('TABLERO VISIBLE')
                            print(tablero_visible)
                            print('\n')
                            time.sleep(1)
                            turno_jugador = False
                            break
                    else:
                        print("Coordenadas fuera de rango. Vuelve a intentarlo.")
                        turno_jugador = False            
            else:
                print('¡HAS PERDIDO!')
                juego_en_progreso=False
                break

        if turno_jugador == False and juego_en_progreso==True:
            contador_turnos_maquina = contador_turnos_maquina +1
            print(f'el turno actual de la maquina es: {contador_turnos_maquina}')

            if 'O' in tablero:
                disparo_maquina_x = np.random.randint(0,10)
                disparo_maquina_y = np.random.randint(0,10)
                
                if tablero[disparo_maquina_x, disparo_maquina_y] == "O":
                    tablero[disparo_maquina_x, disparo_maquina_y] = "X"
                    print("¡TE HAN DADO!")
                    for barco in var.barcos_lista:
                            if all(coordinate == "X" for coordinate in barco.coord):
                                print(f"{barco.nombre} HUNDIDO!\n")
                    print('TABLERO DEL JUGADOR')
                    print(tablero)
                    print('\n')
                    time.sleep(1)
                    turno_jugador = False

                elif tablero[disparo_maquina_x, disparo_maquina_y] == " ":
                    tablero[disparo_maquina_x, disparo_maquina_y] = "-"
                    print("LA MÁQUINA HA DISPARADO AL AGUA!")
                    print('TABLERO DEL JUGADOR')
                    print(tablero)
                    print('\n')
                    time.sleep(1)
                    turno_jugador = True
                    
                elif tablero[disparo_maquina_x, disparo_maquina_y] == "X":
                    print("Disparo previamente realizado por la máquina!")
                    print('TABLERO DEL JUGADOR')
                    print(tablero)
                    print('\n')
                    time.sleep(1)
                    turno_jugador = False

                elif tablero[disparo_maquina_x, disparo_maquina_y] == "-":
                    print("Disparo previamente realizado por la máquina!")
                    print('TABLERO DEL JUGADOR')
                    print(tablero)
                    print('\n')
                    time.sleep(1)
                    turno_jugador = False

            else:
                print('¡HAS GANADO!')
                juego_en_progreso=False
                break
import HF_fun_m as fun
import HF_var_m as var

# MENSAJES DE INICIO
print ("Bienvenido a hundir la flota!!")
nombre_jugador = input("¿Como te llamas?")
print(f"Maravilloso! {nombre_jugador}, seleciona a continuación el nivel de dificultad:")

while True:
    dificultad = input("Normal / No gano ni queriendo")  #Aquí la idea es hacer un if dificultad == "Normal" seguir y con un else hacer la variante dificil

    if dificultad.lower() == "normal": #Ahora mismo solo funciona con dificultad normals
       
    #print(f"Por defecto el tamaño del tablero es de 10x10, ¿Quieres usar estas medidas {nombre_jugador}?")
    #tablero_estandar = input("Si / No").lower() #Aquí habría que hacer otro if para dar otras medidas, aunque podemos pasar y hacerlo 10x10 y ya

        fun.crea_tablero()
        fun.crea_tablero_maquina()

        print(f"Este es tu tablero {nombre_jugador}, ¿Quieres colocar tus barcos aleatoriamente, o manualmente?")
        print(var.tablero)

        while True:
            colocar_barcos_input = input(f"Aleatorio / Manual").lower()

            if colocar_barcos_input == "aleatorio":
                fun.generar_todos_los_barcos3()
                break

            elif colocar_barcos_input == "manual":
                fun.generar_todos_los_barcos(tablero_auto = False)
                break

            else:
                print(f"No te he entendido {nombre_jugador}, escribe <Aleatorio> o <manual>")
        fun.generar_todos_los_barcos_maquina()
        print("Este es tu tablero")
        print(var.tablero)

        print(f"{nombre_jugador} Empieza el juego!")

        fun.dispara_propio(tablero_maquina = var.tablero_maquina, tablero_visible= var.tablero_visible)

        break
    #------------------------------------------------------------------------------------------------------------#
    #SE REPITE LO MISMO PERO CON LA DIFICULTAD AUMENTADA, HAY QUE SUSTITUIR LA FUNCION DISPARO POR OTRA MODIFICADA
    #------------------------------------------------------------------------------------------------------------#
    elif dificultad.lower() == "no gano ni queriendo":
        fun.crea_tablero()
        fun.crea_tablero_maquina()

        print(f"Este es tu tablero {nombre_jugador}, ¿Quieres colocar tus barcos aleatoriamente, o manualmente?")
        print(var.tablero)

        while True:
            colocar_barcos_input = input(f"Aleatorio / Manual").lower()

            if colocar_barcos_input == "aleatorio":
                fun.generar_todos_los_barcos()
                break

            elif colocar_barcos_input == "manual":
                fun.generar_todos_los_barcos(tablero_auto = False)
                break

            else:
                print(f"No te he entendido {nombre_jugador}, escribe <Aleatorio> o <manual>")
        fun.generar_todos_los_barcos_maquina()

        print(var.tablero)

        print(f"{nombre_jugador} Empieza el juego!")

        fun.dispara_propio(tablero_maquina = var.tablero_maquina, tablero_visible= var.tablero_visible) #--->SUSTITUIR POR FUN DIFICIL
        break
    #-------------------------------------------------------------------------------------------------#
    #SE REPITE EL BUCLE SI EL JUGADOR NO SABE ESCRIBIR
    #-------------------------------------------------------------------------------------------------#
    else:
        print(f"{nombre_jugador} no te he entendido, escribe <Normal> / <No gano ni queriendo>")


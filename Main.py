import HF_fun as fun

tablero_visible = fun.tablero_visible

print ("Bienvenido a hundir la flota!!")
nombre_jugador = input("¿Como te llamas?")

print(f"Maravilloso! {nombre_jugador}, seleciona a continuación el nivel de dificultad:")
dificultad = input("Normal / No gano ni queriendo")  #Aquí la idea es hacer un if dificultad == "Normal" seguir y con un else hacer la variante dificil

if dificultad.lower() == "normal": #Ahora mismo solo funciona con dificultad normals

    print(f"Por defecto el tamaño del tablero es de 10x10, ¿Quieres usar estas medidas {nombre_jugador}?")
    tablero_estandar = input("Si / No").lower() #Aquí habría que hacer otro if para dar otras medidas, aunque podemos pasar y hacerlo 10x10 y ya

    fun.crea_tablero()
    fun.crea_tablero_maquina()

    print(f"Este es tu tablero {nombre_jugador}, ¿Quieres colocar tus barcos aleatoriamente, o manualmente?")
    print(fun.tablero)

    colocar_barcos_input = input(f"Aleatorio / Manual").lower()

    if colocar_barcos_input == "aleatorio":
        fun.generar_todos_los_barcos()
        

    else:
        fun.generar_todos_los_barcos(tablero_auto = False)
       

    fun.generar_todos_los_barcos_maquina()

    print(fun.tablero)

    print(f"{nombre_jugador} Empieza el juego!")

    fun.dispara_propio(tablero_maquina = fun.tablero_maquina, tablero_visible= fun.tablero_visible)

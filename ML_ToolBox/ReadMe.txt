# Paquete de Funciones para Machine Learning

Este paquete contiene varias funciones útiles para el análisis de datos utilizando la librería Pandas en Python. Estas funciones abordan diferentes aspectos del análisis de datos, como la descripción de un DataFrame, 
la tipificación de variables, la selección de características para problemas de regresión, y la visualización de datos para análisis en regresión.

## Funciones Disponibles

### describe_df

Esta función proporciona información detallada sobre un DataFrame, incluyendo el tipo de datos de cada columna, el porcentaje de valores nulos, los valores únicos y la cardinalidad.

### tipifica_variables

Esta función tipifica las columnas de un DataFrame en función de su cardinalidad, ayudando a identificar si una variable es categórica o numérica.

### get_features_num_reggresion

Selecciona las características numéricas más relevantes para un problema de regresión en función de su correlación con la variable objetivo.

### plot_features_num_regression

Genera un pairplot para características numéricas basado en su correlación con la variable objetivo.

### get_features_cat_regression

Identifica las variables categóricas que se consideran características importantes para un problema de regresión en función de su correlación con la variable objetivo.

### plot_features_cat_regression

Dibuja histogramas de la variable objetivo para cada característica categórica, permitiendo visualizar su relación.

## Ejemplos de Uso

Aquí hay ejemplos de cómo utilizar algunas de estas funciones:

```python
import pandas as pd
import numpy as np
from ML_ToolBox import describe_df, tipifica_variables

# DataFrame de ejemplo
data = {
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [True, False, True, False, True]
}
df = pd.DataFrame(data)

# Describir el DataFrame
description = describe_df(df)
print(description)

# Tipificar las variables del DataFrame
types = tipifica_variables(df, umbral_cat=5, umbral_con=10.0)
print(types)
```

# Requisitos
Este paquete requiere Python 3.x y las siguientes bibliotecas:

Pandas
NumPy
Seaborn
Matplotlib
SciPy
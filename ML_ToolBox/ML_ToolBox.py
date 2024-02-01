import pandas as pd

# Describe_df 

def describe_df(df):
    """
    Descripción:
        Esta función dá información sobre un DataFrame

    Argumentos:
        df (Pandas Dataframe) : El DataFrame sobre el que quieras obtener información. 

    Returns:
        Pandas Dataframe : Un nuevo DataFrame con las columnas del DataFrame original, el tipo de datos de cada columna, su porcentaje de nulos o misings, sus valores únicos y su porcentaje de cardinalidad.
    """
    # Columnas
    columns_names = df.columns.tolist()

    # Tipo de datos
    data_type = df.dtypes

    # Porcentaje de nulos/missings
    missings = (df.isnull().mean() * 100).round(2)
    
    # Valores únicos:
    unique_values = df.nunique()

    # Cardinalidad:
    card = (unique_values / len(df) * 100).round(2)

    # Dataframe resultado:
    new_df = pd.DataFrame({
        "Data_Type" : data_type,
        "Missings(%)" : missings,
        "Unique_Values" : unique_values,
        "Card(%)" : card
    })
    new_df.index = columns_names

    new_df = new_df.T

    return new_df

# Tipificar variables:

def tipifica_variables(df, umbral_cat, umbral_con):
    """
    Descripción:
    Esta función tipifica las diferentes columnas de un DF según su porcentaje de cardinalidad.

    Argumentos:
        df (Pandas Dataframe): Dataframe que quieras tipificar sus columnas.
        umbral_cat (INT): Porcentaje de cardinalidad para que el tipo sea considerado categórico.
        umbral_con (FLOAT): Porcentaje de cardinalidad para que el tipo pase de númerico discreto a numérico continuo.

    Returns:
        Pandas Dataframe: Un Dataframe cuyas filas son las columnas del DataFrame original que devuelve en una nueva columna el tipo de variable según los umbrales escogidos
    """
    tipo_variable = []

    # Cardinalidad de las columnas:
    for col in df.columns:
        card = (df[col].nunique() / len(df) * 100)

        # Sugerencias de tipo:
        if card == 2:
            tipo = "Binaria"
        elif card < umbral_cat:
            tipo = "Categorica"
        elif card >= umbral_con:
            tipo = "Numerica Continua"
        else: tipo = "Numerica Discreta"

        tipo_variable.append({
            "Variable" : col,
            "Tipo" : tipo
        })

        # Nuevo DF:
    df_type = pd.DataFrame(tipo_variable)

    return df_type

# Get Features Num Regression


# Plot Features Num Regression


# Get Features Cat Regression


# Plot Features Cat Regression
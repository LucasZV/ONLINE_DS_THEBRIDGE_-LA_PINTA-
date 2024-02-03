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
def get_features_num_reggresion(df,target_col, umbral_corr,pvalue=None):
    if target_col not in df.columns:
        """
    get_features_num_regresion: selecciona las características numéricas para un problema de regrersión. 
    Esta función verifica que los datos sean adecuados y automáticamente selecciona las columnas numéricas que
    están más relacionadas con la que estamos tratando. Sirve para hacer predicciones más precisas.

    Argumentos:
     - df (DataFrame): El conjunto de datos.
     - target_col (str): El nombre de la columna objetivo que queremos predecir.
     - umbral_corr (float): Umbral de correlación para seleccionar características (entre 0 y 1).
     - pvalue (float, opcional): Umbral de significación estadística (entre 0 y 1) para el test de correlación. Por defecto, None.

    Retorna:
    - selected_features (list): Lista de características seleccionadas que cumplen con los criterios.
    """  
    
    if not np.issubdtype(df[target_col].dtype, np.number):# Comprobar que la columna objetivo es una variable numérica continua
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    
    if not (0 <= umbral_corr <= 1): # Comprobar que el umbral de correlación está entre 0 y 1
        print("Error: umbral_corr debe estar entre 0 y 1")
        return None 
    
    if pvalue is not None and not (0 <= pvalue <= 1):# Comprobar que pvalue es None o un número entre 0 y 1
        print("Error: pvalue debe ser None o un número entre 0 y 1.")
        return None
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()  # Obtener columnas numéricas del DataFrame

    Correlations = []  # Calcular la correlación y p-values para cada columna numérica con 'target_col'
    for col in numeric_cols:
        if col != target_col:
            correlation, p_value = pearsonr(df[col], df[target_col])
            Correlations.append((col, correlation, p_value))

    Selected_features = []  # Filtrar las columnas basadas en el umbral de correlación y p-value
    for col, corr, p_value in Correlations:
        if abs(corr) > umbral_corr and (pvalue is None or p_value <= (1 - pvalue)):
            Selected_features.append(col)
    return Selected_features



# Plot Features Num Regression
def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera un pairplot para características numéricas basado en la correlación con la columna objetivo.

    Parámetros:
    - dataframe (pd.DataFrame): DataFrame de entrada.
    - target_col (str, opcional): Columna objetivo para el análisis de correlación.
    - columns (lista de str, opcional): Lista de columnas para el pairplot. Por defecto, es una lista vacía.
    - umbral_corr (float, opcional): Umbral de correlación. Por defecto, es 0.
    - pvalue (float o None, opcional): Valor p para el test de significación estadística de la correlación. Por defecto, es None.

    Retorna:
    - lista de str: Lista de columnas que cumplen con las condiciones específicas.
    """

    # Valoresde entrada
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas.")

    if target_col not in dataframe.columns:
        raise ValueError("La columna objetivo '{}' no se encuentra en el dataframe.".format(target_col))

    if not isinstance(columns, list):
        raise ValueError("'columns' debe ser una lista de strings.")

    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Todas las columnas en 'columns' deben existir en el dataframe.")

    if not isinstance(umbral_corr, (int, float)):
        raise ValueError("'umbral_corr' debe ser un valor numérico.")

    if not (isinstance(pvalue, (float, int)) or pvalue is None):
        raise ValueError("'pvalue' debe ser un valor numérico o None.")

    if not columns: # Si 'columns' está vacío, utiliza todas las columnas numéricas
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    selected_columns = []  # Filtra las columnas basadas en la correlación y el valor p
    for col in columns:
        if col != target_col:
            correlation, p_val = pearsonr(dataframe[col], dataframe[target_col])

            if abs(correlation) > umbral_corr and (pvalue is None or p_val < pvalue):
                selected_columns.append(col)



    #  pairplot para las columnas seleccionadas
    if selected_columns:
        num_plots = len(selected_columns) // 4 + 1
        for i in range(num_plots):
            start_index = i * 4
            end_index = (i + 1) * 4
            subset_columns = [target_col] + selected_columns[start_index:end_index]
            
            subset_data = dataframe[subset_columns]
            sns.pairplot(subset_data, diag_kind='kde')
            plt.show()

    return selected_columns


# Get Features Cat Regression


# Plot Features Cat Regression
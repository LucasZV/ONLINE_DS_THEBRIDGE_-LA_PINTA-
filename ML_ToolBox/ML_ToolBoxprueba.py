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

    selected_features = []  # Filtrar las columnas basadas en el umbral de correlación y p-value
    for col, corr, p_value in Correlations:
        if abs(corr) > umbral_corr and (pvalue is None or p_value <= (1 - pvalue)):
            selected_features.append(col)

    if not selected_features: #Si no tiene correlación nos imprime este mensaje:
                 return "No hay correlación"        

    return selected_features



def plot_features_num_regression(dataframe, target_col="", columns=[]):
    """
    Genera un pairplot para características numéricas basado en la correlación con la columna objetivo.

    Parámetros:
    - dataframe (pd.DataFrame): DataFrame de entrada.
    - target_col (str): Columna objetivo para el análisis de correlación.
    - columns (lista de str): Lista de columnas para el pairplot.

    Retorna:
    - None
    """
    # Verifica que el argumento 'dataframe' sea un DataFrame de pandas.
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas.")

    # Verifica que 'target_col' esté en las columnas del DataFrame.
    if target_col not in dataframe.columns:
        raise ValueError("La columna objetivo '{}' no se encuentra en el dataframe.".format(target_col))

    # Verifica que 'columns' sea una lista de strings.
    if not isinstance(columns, list):
        raise ValueError("'columns' debe ser una lista de strings.")

    # Verifica que todas las columnas en 'columns' existan en el DataFrame.
    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Todas las columnas en 'columns' deben existir en el dataframe.")

    # Si 'columns' está vacío, utiliza todas las columnas numéricas.
    if not columns:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()

    # Imprime la correlación y p-values entre cada columna numérica y 'target_col'.
    for col in columns:
        if col != target_col:
            correlation, p_val = pearsonr(dataframe[col], dataframe[target_col])
            print(f"Correlación entre {col} y {target_col}: {correlation:.4f}, p-value: {p_val:.4f}")

    # Genera un pairplot para las columnas seleccionadas.
    subset_columns = [target_col] + columns
    subset_data = dataframe[subset_columns]
    sns.pairplot(subset_data, diag_kind='kde')
    plt.show()

    # Ejemplo de uso
plot_features_num_regression(df, target_col='', columns=[''])



# Get Features Cat Regression


# Plot Features Cat Regression
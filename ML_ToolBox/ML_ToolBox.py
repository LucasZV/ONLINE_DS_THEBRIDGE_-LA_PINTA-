import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency

# Describe_df 

def describe_df(df : pd.DataFrame):
    """
    Descripción:
        Esta función dá información sobre un DataFrame

    Argumentos:
        df (Pandas Dataframe) : El DataFrame sobre el que quieras obtener información. 

    Returns:
        Pandas Dataframe : Un nuevo DataFrame con las columnas del DataFrame original, el tipo de datos de cada columna, su porcentaje de nulos o misings, sus valores únicos y su porcentaje de cardinalidad.
    """
    #Comprobar que el dataframe pasado es un dataframe
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento 'df' debe ser un Dataframe de Pandas.")
    
    # Filtrar valores 'unk' y 'unknown' para el cálculo de tipos de datos
    df_filtered = df.replace(['UNK',"unk", "-", "unknow", "unknown" "missing", "nan", "NaN"], np.nan)

    # Columnas
    columns_names = df.columns.tolist()

    # Tipo de datos
    data_type =  df_filtered.apply(lambda x: x.dropna().dtype)

    # Porcentaje de nulos/missings
    missings = ((df_filtered.isnull() | df_filtered.isin(['unknown', "unknow", 'unk', "UNK", '-', 'missing', "NaN", "NAN", "nan"]) | df_filtered.isna()).mean() * 100).round(2)
    
    # Valores únicos:
    unique_values = df_filtered.nunique()

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

def tipifica_variables(df, umbral_cat: int, umbral_con: float):
    """
    Descripción:
    Esta función tipifica las diferentes columnas de un DF según su cardinalidad.

    Argumentos:
        df (Pandas Dataframe): Dataframe que quieras tipificar sus columnas.
        umbral_cat (INT): Cardinalidad para que el tipo sea considerado categórico.
        umbral_con (FLOAT): Cardinalidad para que el tipo pase de numérico discreto a numérico continuo.

    Returns:
        Pandas Dataframe: Un Dataframe cuyas filas son las columnas del DataFrame original que devuelve en una nueva columna el tipo de variable según los umbrales escogidos
    """
    # Validar los tipos de los argumentos
    if not isinstance(umbral_cat, int):
        raise ValueError("El umbral para variables categóricas (umbral_cat) debe ser un INT.")
    if not isinstance(umbral_con, float):
        raise ValueError("El umbral para variables continuas (umbral_con) debe ser un FLOAT.")

    tipo_variable = []

    # Cardinalidad de las columnas:
    for col in df.columns:
        # Verificar si la columna es numérica
        if pd.api.types.is_numeric_dtype(df[col]):
            card = df[col].nunique()

            # Sugerencias de tipo:
            if card == 2:
                tipo = "Binaria"
            elif card < umbral_cat:
                tipo = "Categórica"
            elif card >= umbral_con:
                tipo = "Numérica Continua"
            else:
                tipo = "Numérica Discreta"

            tipo_variable.append({
                "Variable": col,
                "Tipo": tipo
            })
        else:
            print(f"La columna {col} no es numérica.")

    # Nuevo DF:
    df_type = pd.DataFrame(tipo_variable)

    return df_type

# Ejemplo de uso:
# result = tipifica_variables(df, umbral_cat=5, umbral_con=10.0)

# Get Features Num Regression
def get_features_num_reggresion(df,target_col, umbral_corr,pvalue=None):
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
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no esta en el dataframe")

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
    return selected_features

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
def get_features_cat_regression(df,target_col,pvalue=0.05,umbral_card=0.5):
    """
    Descripción:
        La función identifica las variables categóricas de un dataframe que se consideran features de una variable target en función de su correlación.

    Argumentos:
        df (Pandas Dataframe) : El DataFrame sobre el que trabajar.
        target_col: varible objetivo (numérica continua o discreta) del df.
        pvalue: valor que restado a 1 nos indica el intervalo de confianza para la identificación de features (cómo correlan con la vasriable target) 
        umbral_card: umbral de cardinalidad.

    Returns:
        cat_features: lista de las variables categóricas que han sido identificadas como features.
    """
    umbral_card=0.5
    if target_col not in df.columns:
        print(f"Error: la columna {target_col} no existe.")
        return None
    # martin
    if not np.issubdtype(df[target_col].dtype, np.number) or ((df.nunique()/len(df)*100) < umbral_card):
        print(f"Error: la columna {target_col} no es numérica y/o su cardinalidad es inferior a {umbral_card}.")
        return None
    
    """ if target_col.dtype() not in [int,float] or ((df.nunique()/len(df)*100) < umbral_card):
        print(f"Error: la columna {target_col} no es numérica y/o su cardinalidad es inferior a {umbral_card}.")
        return None """
    
    if pvalue.dtype() != float:
        print(f"Error: la variable {pvalue} no es float.")
        return None
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    cat_features = []
    for col in categorical_cols: # Calcular la correlación y p-values para cada columna numérica con 'target_col'
        contingency_table = pd.crosstab(df[col], df[target_col])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < pvalue:
            cat_features.append(col)
    return(cat_features)

# Plot Features Cat Regression
def plot_features_cat_regression(df,target_col="",columns=[],pvalue=0.05,with_individual_plot=False, umbral_card=0.5):
    """
    Descripción:
        La función dibuja los histogramas de la variable objetivo para cada una de las features.

    Argumentos:
        df: El DataFrame sobre el que trabajar.
        target_col: varible objetivo (numérica continua o discreta) del df.
        columns: listado de variables categóricas. Por defecto está vacío.
        pvalue: valor que restado a 1 nos indica el intervalo de confianza para la identificación de features (cómo correlan con la variable target). Por defecto 0.05.
        with_individual_plot: argumento para dibujar el histograma individual o agrupado (por defecto).
        umbral_card: umbral de cardinalidad.

    Returns:
        figure: histogramas
    """

    selected_features=get_features_num_reggresion(df,target_col, umbral_corr=0.5)
    cat_features=get_features_cat_regression(df,target_col,pvalue=0.05,umbral_card=0.5)
    # Comprobación de los valores de entrada:
    # variable objetivo incluída en el df (columna)
    if target_col not in df.columns:
        print(f"Error: la columna {target_col} no existe.")
        return None
    # variable objetivo numérica y con alta cardinalidad (superior al umbral deseado)
    if (target_col.dtype() not in [int,float]) or (df.nunique()/len(df)*100<umbral_card):
        print(f"Error: la columna {target_col} no es numérica y/o su cardinalidad es inferior a {umbral_card}.")
        return None
    # argumento pvalue decimal
    if pvalue.dtype() != float:
        print(f"Error: la variable {pvalue} no es float.")
        return None
    
    # Una vez comprobados los valores de entrada, revisamos el listado de columnas:
    # Si es una lista vacía (por defecto), se cogen las features numéricas del df y se dibujan los histogamas para la variable objetivo y cada feature numérica.
    if columns==[]:
        columns=selected_features
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=target_col, hue=feature, bins=20, multiple="stack")
            plt.xlabel(target_col)
            plt.ylabel('Frecuencia')
            plt.title(f'Histogramas agrupados de {target_col} para features en {feature}')
            plt.show()
     # Si la lista viene informada, se comprueba que sean features categóricas del df y se dibujan los histogamas para la variable objetivo y cada feature categórica.
    else:
        cat_features_hist=[]
        for col in columns:
            if col in cat_features:
                cat_features_hist.append(col)
        for feature in cat_features_hist:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=feature, hue=target_col)
            plt.xlabel(feature)
            plt.ylabel('Conteo')
            plt.title(f'Gráfico de barras agrupado para {target_col} en función de {feature}')
            plt.legend(title=target_col)
            plt.show()

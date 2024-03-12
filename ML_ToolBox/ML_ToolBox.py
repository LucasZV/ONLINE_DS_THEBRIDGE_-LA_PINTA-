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

#ejemplo de uso
# get_features_num_reggresion(df, 'target_column', 0.5, pvalue=0.05)



# Plot Features Num Regression
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
# plot_features_num_regression(df, target_col='', columns=[''])


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
    
    if not np.issubdtype(df[target_col].dtype, np.number) or (not any(df[col].nunique() / len(df) * 100 > umbral_card for col in df.columns)):
        print(f"Error: la columna {target_col} no es numérica y/o su cardinalidad es inferior a {umbral_card}.")
        return None
    
    if not isinstance(pvalue, float):
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
def plot_features_cat_regression(df,target_col="",columns=[],umbral_card=0.5,pvalue=0.05,with_indivudual_plot=False,umbral_corr=0):
    """
    Descripción:
        La función dibuja los histogramas de la variable objetivo para cada una de las features.

    Argumentos:
        df: El DataFrame sobre el que trabajar.
        target_col: varible objetivo (numérica continua o discreta) del df.
        columns: listado de variables categóricas. Por defecto está vacío.
        pvalue: valor que restado a 1 nos indica el intervalo de confianza para la identificación de features (cómo correlan con la variable target). Por defecto 0.05.
        with_individual_plot: argumento para dibujar el histograma individual o agrupado (por defecto).

    Returns:
        figure: histogramas
    """
    # Comprobación de los valores de entrada:
    if target_col not in df.columns:
        print(f"Error: la columna {target_col} no existe.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    if (df[target_col].nunique()/(len(df[target_col])*100))<umbral_card:
        print(f"Error: la cardinalidad de la columna {target_col} es inferior a {umbral_card}.")
        return None
    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue debe ser None o un número entre 0 y 1.")
        return None
    
    
    # Revisión del listado columns y creación de los gráficos.
    # Si columns está vacía:
    if columns==[]:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        num_features = []  # Filtra las columnas basadas en la correlación y el valor p
        for col in columns:
            if col != target_col:
                correlation, p_value = pearsonr(df[col], df[target_col])
                if abs(correlation) > umbral_corr and (pvalue is None or p_value < pvalue):
                    num_features.append(col)
        num_cols = len(num_features)
        fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5))
        for i, col in enumerate(num_features):
            ax = axes[i]
            sns.histplot(data=df, x=target_col, y=col, ax=ax, bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f'{col} vs {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()   
    # Si columns contiene info:
    else:
        columns_in_df=[]
        for col in columns:
            if col in df.columns:
                columns_in_df.append(col)
        if columns_in_df==[]:
            print(f"Error: las columnas no coinciden con las del df.")
            return None                
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        cat_features = []
        for col in categorical_cols: 
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_value, dof, expected= chi2_contingency(contingency_table)
            if p_value < pvalue:
                cat_features.append(col)
        if not cat_features:
            print("No se encontraron características categóricas significativas.")
            return None
        for col in columns_in_df:
            if col in cat_features:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=target_col, hue=col, palette='Set1', multiple='stack')
                plt.title(f'Histograma de {col} en relación con {target_col}')
                plt.xlabel(target_col)
                plt.ylabel('Frecuencia')
                plt.legend(title=col)
                plt.show()
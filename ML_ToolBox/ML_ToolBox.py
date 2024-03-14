import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency

# Importo librerias
import pandas as pd
import ML_ToolBox as tbox
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import pearsonr, chi2_contingency
from sklearn.feature_selection import SelectFromModel

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
                for valor in df[col].unique():
                    sns.histplot(df[df[col] == valor][target_col], kde= True, label  = str(valor))
                plt.title(f'Histograma de {col} en relación con {target_col}')
                plt.xlabel(target_col)
                plt.ylabel('Frecuencia')
                plt.legend(title=col)
                plt.show()


def super_selector(dataset, target_col="", selectores=None, hard_voting=[]):
    """
    Función para realizar la selección de características en un conjunto de datos.

    Parámetros:
    - dataset (pd.DataFrame): El conjunto de datos de entrada.
    - target_col (str): El nombre de la columna objetivo. Debe ser una columna válida en el dataframe. Si es una cadena vacía, se omite.
    - selectores (dict): Un diccionario que especifica los métodos de selección de características a aplicar y sus parámetros. Las claves pueden ser "KBest", "FromModel", "RFE" o "SFS".
    - hard_voting (list): Una lista de características para incluir. Si está vacía, se omite.

    Retorna:
    - dict: Un diccionario que contiene las listas de características seleccionadas para cada método en selectores y una lista adicional para el hard voting.

    # Ejemplo de uso
    selectores_ejemplo = {
        "KBest": 5,
        "FromModel": [RandomForestClassifier(), 5],
        "RFE": [LogisticRegression(), 5, 1],
        "SFS": [RandomForestClassifier(), 3]
    }

    # Seleccionar características usando super_selector
    resultado_ejemplo = super_selector(train_set_titanic, target_col="Survived", selectores=selectores_ejemplo, hard_voting=["Pclass", "who", "embarked_S", "fare", "age"])

    # Imprimir el resultado
    print("Resultado del ejemplo:")
    for key, value in resultado_ejemplo.items():
        print(f"{key}: {value}")
    """
    result_dict = {}
    
    # Verificar si target_col es válido
    if target_col and target_col not in dataset.columns:
        raise ValueError(f"'{target_col}' no es una columna válida en el dataframe.")
    
    # Obtener las columnas no numéricas
    columnas_no_numericas = dataset.select_dtypes(exclude=['int', 'float']).columns.tolist()
    
    # Iterar sobre cada columna no numérica y transformarla usando LabelEncoder
    # Crear una instancia de LabelEncoder
    label_encoder = LabelEncoder()
    for col in columnas_no_numericas:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    ## Obtener todas las columnas numéricas y la columna objetivo
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir la columna objetivo si está presente en las columnas numéricas
    if target_col in numeric_columns:
        numeric_columns.remove(target_col)

    selected_features = numeric_columns

    # Comprobar selectores
    if not selectores or selectores is None:
        # Si no hay selectores
        selected_features = [col for col in dataset.columns if
                            len(dataset[col].unique()) != 1 and # cardinalidad distinta de 1
                            (len(dataset[col].unique()) / len(dataset)) * 100 != 100] # Cardinalidad distinta del 100%, no índices
        selected_features.remove(target_col)
        result_dict["default"] = selected_features

    # Aplicar selectores      
    else:
        for key, value in selectores.items():

            if key == "KBest":

                kbest_selector = SelectKBest(f_classif, k=value)
                kbest_selector.fit(dataset[numeric_columns], dataset[target_col])
                kbest_features = list(np.array(numeric_columns)[kbest_selector.get_support()])
                result_dict["KBest"] = kbest_features

            elif key == "FromModel":
                if len(value) >= 2:
                    model, threshold = value[0], value[1]
                    if isinstance(threshold, int) and threshold > len(numeric_columns):
                        raise ValueError(f"El umbral no puede ser mayor que el número total de columnas numéricas. --> {len(numeric_columns)}")
                    # Verificar si threshold es válido
                    # Si threshold es median o mean:
                    elif isinstance(threshold, str) and threshold.lower() in ['median', 'mean']: # Calcular el umbral basado en la mediana o la media de las importancias
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            if threshold.lower() == 'median':
                                threshold = np.median(importances)
                            elif threshold.lower() == 'mean':
                                threshold = np.mean(importances)
                            else:
                                raise ValueError(f"Valor no válido para 'threshold': {threshold}")
                        else:
                            # Utilizar get_support() si no hay feature_importances_
                            model.fit(dataset[numeric_columns], dataset[target_col])  # Ajustar el modelo
                            sfm_selector = SelectFromModel(model, threshold=threshold)
                            sfm_selector.fit(dataset[numeric_columns], dataset[target_col])
                            sfm_features = numeric_columns.copy()
                            sfm_features = list(np.array(sfm_features)[sfm_selector.get_support()])
                            result_dict["FromModel"] = sfm_features
                    # Si es un escalar por median o mean:
                    elif isinstance(threshold, str) and '*' in threshold: # Manejar el caso con un factor de escala
                        scaling_factor = float(threshold.split('*')[0])
                        base_threshold = threshold.split('*')[1].lower()
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            if base_threshold == 'mean':
                                threshold = scaling_factor * np.mean(importances)
                            elif base_threshold == 'median':
                                threshold = scaling_factor * np.median(importances)
                            else:
                                raise ValueError(f"Valor no válido para 'threshold': {threshold}")
                        else:
                            # Utilizar get_support() si no hay feature_importances_
                            model.fit(dataset[numeric_columns], dataset[target_col])  # Ajustar el modelo
                            sfm_selector = SelectFromModel(model, threshold=threshold)
                            sfm_selector.fit(dataset[numeric_columns], dataset[target_col])
                            sfm_features = numeric_columns.copy()
                            sfm_features = list(np.array(sfm_features)[sfm_selector.get_support()])
                            result_dict["FromModel"] = sfm_features
                    elif not isinstance(threshold, (int, float)):
                        raise ValueError(f"'threshold' debe ser un número, 'median', 'mean' o una cadena que siga el formato 'factor*mean', pero se proporcionó: {threshold}")
                    # Si es entero:
                    elif isinstance(threshold, int):
                        model.fit(dataset[numeric_columns], dataset[target_col])  # Ajustar el modelo
                        sfm_selector = SelectFromModel(model, max_features=threshold, threshold=-np.inf)
                    # Si es float:
                    elif isinstance(threshold, (float, float)):
                        model.fit(dataset[numeric_columns], dataset[target_col])  # Ajustar el modelo
                        sfm_selector = SelectFromModel(model, threshold=threshold)
                    else:
                        raise ValueError(f"'threshold' debe ser un entero o un valor compatible con SelectFromModel, pero se proporcionó: {threshold}")
                else:
                    raise ValueError("La lista de 'FromModel' debe contener al menos dos elementos: el modelo y el umbral.")
    
                sfm_selector.fit(dataset[numeric_columns], dataset[target_col])
                sfm_features = numeric_columns.copy()
                sfm_features = list(np.array(sfm_features)[sfm_selector.get_support()])
                result_dict["FromModel"] = sfm_features

 
            elif key == "RFE":
                model, n_features_to_select, step = value[0], value[1], value[2]
                # Verificar si n_features_to_select es válido
                # Verificar si n_features es None:
                if n_features_to_select is None:
                    n_features_to_select = len(numeric_columns) // 2  # Si es None, seleccionar la mitad de las características
                # Verificar si n_features es entero:
                elif isinstance(n_features_to_select, int):
                    n_features_value = n_features_to_select
                # Verificar si n_features es float:
                elif isinstance(n_features_to_select, (float, float)) and 0 <= n_features_to_select <= 1:
                    n_features_value = int(n_features_to_select * len(numeric_columns))
                else:
                    raise ValueError(f"'n_features_to_select' debe ser un entero, un float entre 0 y 1, o None, pero se proporcionó: {n_features_to_select}")
                # Verificar si step es válido
                # Verifica si step es int:
                if isinstance(step, int) and step >= 1:
                    step_value = step
                # Verifica si step es float:
                elif isinstance(step, (float, float)) and 0 < step < 1:
                    step_value = int(step * len(numeric_columns))
                else:
                    raise ValueError(f"'step' debe ser un entero >= 1 o un float entre 0 y 1, pero se proporcionó: {step}")
                rfe_selector = RFE(model, n_features_to_select=n_features_to_select, step=step_value)
                rfe_selector.fit(dataset[numeric_columns], dataset[target_col])
                rfe_features = numeric_columns.copy()
                rfe_features = list(np.array(rfe_features)[rfe_selector.support_])
                result_dict["RFE"] = rfe_features

            elif key == "SFS":
                model, n_features = value[0], value[1]
                # Verificar si n_features es válido
                if n_features == "auto":
                    n_features_value = len(numeric_columns) // 2  # Mitad del número total de características
                elif isinstance(n_features, int):
                    n_features_value = n_features
                elif isinstance(n_features, (float, float)) and 0 <= n_features <= 1:
                    n_features_value = int(n_features * len(numeric_columns))
                else:
                    raise ValueError(f"'n_features' debe ser 'auto', un entero o un float entre 0 y 1, pero se proporcionó: {n_features}")
                
                sfs_selector = SequentialFeatureSelector(model, n_features_to_select=n_features_value, direction="forward")
                sfs_selector.fit(dataset[numeric_columns], dataset[target_col])
                sfs_features = numeric_columns.copy()
                if hasattr(sfs_selector, 'k_feature_names_'):
                    sfs_features = list(sfs_selector.k_feature_names_)
                elif hasattr(sfs_selector, 'support_'):
                    sfs_features = list(np.array(numeric_columns)[sfs_selector.support_])
                result_dict["SFS"] = sfs_features
    # Hard Voting
    # Obtener todas las características seleccionadas
    all_selected_features = [result_dict[key] for key in result_dict]
    all_selected_features += [hard_voting] if hard_voting else []

    # Verificar si hay al menos una matriz para concatenar
    if all_selected_features:
        # Realizar la concatenación y realizar el hard voting
        voting_result = [feature for feature, count in Counter(np.concatenate(all_selected_features)).items() if count >= len(all_selected_features) // 2]
    else:
        # Si no hay características seleccionadas, asignar una lista vacía a voting_result
        voting_result = []

    result_dict["hard_voting"] = voting_result

    return result_dict
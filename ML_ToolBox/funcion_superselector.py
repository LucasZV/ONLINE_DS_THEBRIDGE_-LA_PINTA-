# Funcion super_selector: Martin

from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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

    # Crear lista de todas las columnas excluyendo target_col
    all_features = [col for col in dataset.columns if col != target_col]

    # Comprobar selectores
    if not selectores or selectores is None:
        # Si no hay selectores
        selected_features = [col for col in all_features if
                            len(dataset[col].unique()) != 1 and #cardinalidad distinta de 1
                             (len(dataset[col].unique()) / len(dataset)) *100 != 100] #Cardinalidad distinta del 100%, no indices
        result_dict["default"] = selected_features

    # Aplicar selectores    
    else:
        for key, value in selectores.items():

            if key == "KBest":
                kbest_selector = SelectKBest(stats.f_oneway, k=min(value, len(all_features)))
                kbest_selector.fit(dataset[all_features], dataset[target_col].values.reshape(-1, 1))
                kbest_features = list(np.array(kbest_features)[kbest_selector.get_support()])
                result_dict["KBest"] = kbest_features

            elif key == "FromModel":
                model, threshold = value[0], value[1]
                # Verificar si threshold es válido
                # Si threshold es None:
                if threshold is None: 
                    penalty = getattr(model, 'penalty', None)
                    if penalty == 'l1' or penalty=="Lasso" or (hasattr(model, 'get_params') and model.get_params().get('penalty') == 'l1'):
                        threshold = 1e-5
                    else:
                        threshold = "mean"
                # Si threshold es median o mean:
                elif isinstance(threshold, str) and threshold.lower() in ['median', 'mean']: # Calcular el umbral basado en la mediana o la media de las importancias
                    importances = model.feature_importances_
                    if threshold.lower() == 'median':
                        threshold = np.median(importances)
                    elif threshold.lower() == 'mean':
                        threshold = np.mean(importances)
                    else:
                        raise ValueError(f"Valor no válido para 'threshold': {threshold}")
                # Si es un escalar por median o mean:
                elif isinstance(threshold, str) and '*' in threshold: # Manejar el caso con un factor de escala
                    scaling_factor = float(threshold.split('*')[0])
                    base_threshold = threshold.split('*')[1].lower()
                    importances = model.feature_importances_
                    if base_threshold == 'mean':
                        threshold = scaling_factor * np.mean(importances)
                    elif base_threshold == 'median':
                        threshold = scaling_factor * np.median(importances)
                    else:
                        raise ValueError(f"Valor no válido para 'threshold': {threshold}")
                # Si es un valor distinto:
                elif not isinstance(threshold, (int, float)):
                    raise ValueError(f"'threshold' debe ser un número, 'None', 'median', 'mean' o una cadena que siga el formato 'factor*mean', pero se proporcionó: {threshold}")
                # Si es entero:
                elif isinstance(threshold, int):
                    sfm_selector = SelectFromModel(model, max_features=threshold, threshold=-np.inf)
                # Si es float:
                elif isinstance(threshold, (float, np.float)):
                    sfm_selector = SelectFromModel(model, threshold=threshold)
                else:
                    raise ValueError(f"'threshold' debe ser un entero o un valor compatible con SelectFromModel, pero se proporcionó: {threshold}")

                sfm_selector.fit(dataset[all_features], dataset[target_col])
                sfm_features = all_features.copy()
                sfm_features = list(np.array(sfm_features)[sfm_selector.get_support()])
                result_dict["FromModel"] = sfm_features
 
            elif key == "RFE":
                model, n_features, step = value[0], value[1], value[2]
                # Verificar si n_features_to_select es válido
                # Verificar si n_features es None:
                if n_features is None:
                    n_features_to_select = len(all_features) // 2  # Si es None, seleccionar la mitad de las características
                # Verificar si n_features es entero:
                elif isinstance(n_features, int):
                    n_features_to_select = n_features
                # Verificar si n_features es float:
                elif isinstance(n_features, (float, np.float)) and 0 <= n_features <= 1:
                    n_features_to_select = int(n_features * len(all_features))
                else:
                    raise ValueError(f"'n_features_to_select' debe ser un entero, un float entre 0 y 1, o None, pero se proporcionó: {n_features}")
                # Verificar si step es válido
                # Verifica si step es int:
                if isinstance(step, int) and step >= 1:
                    step_value = step
                # Verifica si step es float:
                elif isinstance(step, (float, np.float)) and 0 < step < 1:
                    step_value = int(step * len(all_features))
                else:
                    raise ValueError(f"'step' debe ser un entero >= 1 o un float entre 0 y 1, pero se proporcionó: {step}")
                rfe_selector = RFE(model, n_features_to_select=n_features, step=step)
                rfe_selector.fit(dataset[all_features], dataset[target_col])
                rfe_features = all_features.copy()
                rfe_features = list(np.array(rfe_features)[rfe_selector.support_])
                result_dict["RFE"] = rfe_features

            elif key == "SFS":
                model, n_features = value[0], value[1]
                 # Verificar si n_features es válido
                if n_features == "auto":
                    n_features_value = "best"
                elif isinstance(n_features, int):
                    n_features_value = n_features
                elif isinstance(n_features, (float, np.float)) and 0 <= n_features <= 1:
                    n_features_value = int(n_features * len(all_features))
                else:
                    raise ValueError(f"'n_features' debe ser 'auto', un entero o un float entre 0 y 1, pero se proporcionó: {n_features}")
                sfs_selector = SequentialFeatureSelector(model, n_features=n_features, forward=True, floating=False,
                                                          scoring='accuracy', cv=0)
                sfs_selector.fit(dataset[all_features], dataset[target_col])
                sfs_features = all_features.copy()
                sfs_features = list(sfs_selector.k_feature_names_)
                result_dict["SFS"] = sfs_features

    # Hard Voting
    all_selected_features = [result_dict[key] for key in result_dict]
    all_selected_features += [hard_voting] if hard_voting else []
    
    voting_result = [feature for feature, count in Counter(np.concatenate(all_selected_features)).items() if count >= len(all_selected_features) // 2]
    result_dict["hard_voting"] = voting_result

    return result_dict
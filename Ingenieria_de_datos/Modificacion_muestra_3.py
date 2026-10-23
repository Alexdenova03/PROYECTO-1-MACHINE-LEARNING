import pandas as pd
import numpy as np
import re

# ==============================================================================
# 1. CARGA DE DATOS ORIGINALES (INMUTABLES)
# ==============================================================================
muestra_3 = pd.read_excel("data/Muestras a Clavetear.xlsx", sheet_name="Muestra_3")
respaldo_control = muestra_3.copy(deep=True)


# ==============================================================================
# 2. FUNCIÓN DE PROCESAMIENTO PURA (SOBRE COPIA)
# ==============================================================================
def limpiar_y_alinear_muestra_3(df_original):
    """
    Genera una copia profunda y aplica únicamente:
    1. Alineación de los 67 registros desfasados (izq y der).
    2. Conversión de 'Ocupantes' a número entero (quitando 'OCUP.').
    3. Eliminación del punto final en 'descripcion'.
    4. Creación de la columna 'Transmisión' extrayendo el último token 
       y mapeándolo al formato del catálogo ('A' para automático, 'S' para estándar).
    """
    df_copia = df_original.copy(deep=True)
    
    # Asegurar cadenas limpias en columnas clave
    df_copia["descripcion"] = df_copia["descripcion"].astype("string").str.strip()
    ocup_str = df_copia["Ocupantes"].astype("string").str.strip()
    mod_str = df_copia["Modelo"].astype("string").str.strip()
    modelo_num = pd.to_numeric(df_copia["Modelo"], errors="coerce")

    # A) Identificación de regímenes desfasados
    mask_der = ocup_str.isin(["PDK AUT.", "AUT.", "STD."]).fillna(False)
    mask_izq = (
        modelo_num.isna()
        & ~mask_der
        & ocup_str.str.fullmatch(r"\d{4}").fillna(False)
    )

    # B) Corrección de Desfase a la Derecha (7 registros)
    descripcion_der = df_copia.loc[mask_der, "descripcion"].fillna("").str.strip()
    df_copia.loc[mask_der, "descripcion"] = (descripcion_der + " " + ocup_str[mask_der]).str.strip()
    df_copia.loc[mask_der, "Ocupantes"] = mod_str[mask_der]
    df_copia.loc[mask_der, "Modelo"] = np.nan

    # C) Corrección de Desfase a la Izquierda (60 registros)
    descripcion_izq = df_copia.loc[mask_izq, "descripcion"]
    ocup_ext = descripcion_izq.str.extract(r"\b(\d+)\s*OCUP\.?", flags=re.IGNORECASE)[0]
    
    df_copia.loc[mask_izq, "descripcion"] = (
        descripcion_izq.str.replace(r"\s*\b\d+\s*OCUP\.?\s*", " ", regex=True, flags=re.IGNORECASE)
        .str.strip()
    )
    df_copia.loc[mask_izq, "Modelo"] = pd.to_numeric(ocup_str[mask_izq], errors="coerce")
    df_copia.loc[mask_izq, "Ocupantes"] = ocup_ext

    # D) Conversión de 'Ocupantes' y 'Modelo' a números enteros limpios (Int64)
    df_copia["Ocupantes"] = pd.to_numeric(
        df_copia["Ocupantes"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce"
    ).astype("Int64")
    
    df_copia["Modelo"] = pd.to_numeric(df_copia["Modelo"], errors="coerce").astype("Int64")

    # E) Extracción de la transmisión desde el último token de 'descripcion'
    # Obtenemos la última palabra (ej. "AUT.", "STD.", "PDK", etc.)
    ultima_palabra = df_copia["descripcion"].str.split().str[-1].str.replace(".", "", regex=False).str.upper()
    
    # Mapeo al formato del catálogo: 'A' para automático/PDK, 'S' para estándar/manual
    def mapear_transmision(token):
        if pd.isna(token):
            return pd.NA
        if any(term in token for term in ["AUT", "PDK", "TIPTRONIC", "CVT", "SELECTSHIFT", "7SPEED"]):
            return "A"
        if any(term in token for term in ["STD", "MAN"]):
            return "S"
        return pd.NA

    df_copia["Transmisión"] = ultima_palabra.apply(mapear_transmision)

    
    # F) Limpieza final de 'descripcion'
    # Conservar la transmisión (AUT, STD, PDK AUT, etc.)
    # Eliminar únicamente el punto final de la descripción

    df_copia["descripcion"] = (
        df_copia["descripcion"]
        .str.replace(r"\.$", "", regex=True)
        .str.strip()
    )

    return df_copia


# ==============================================================================
# 3. EJECUCIÓN Y VALIDACIÓN
# ==============================================================================

muestra_3_homologada = limpiar_y_alinear_muestra_3(muestra_3)

# Definir las columnas que queremos visualizar
columnas_vista = [
    "descripcion",
    "Ocupantes",
    "Modelo",
    "Transmisión"
]

# Mostrar las primeras 12 filas
print("\n=== MUESTRA 3 HOMOLOGADA ===")
print(muestra_3_homologada[columnas_vista].head(12).to_string(index=False))

# Verificar que la muestra original no fue modificada
pd.testing.assert_frame_equal(muestra_3, respaldo_control)

print("\nLa muestra original permanece intacta.")

import runpy
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import re
import unicodedata

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# ==============================================================================
# 1. RECUPERACIÓN DE LA MUESTRA 3 HOMOLOGADA Y DEL CATÁLOGO
# ==============================================================================

raiz_proyecto = Path(__file__).resolve().parent.parent

# Si el archivo se encuentra directamente en la raíz del repositorio,
# utilizamos esa carpeta como raíz.
if not (raiz_proyecto / "Ingenieria_de_datos").is_dir():
    raiz_proyecto = Path(__file__).resolve().parent

ruta_muestra = raiz_proyecto / "Ingenieria_de_datos" / "Modificacion_muestra_3.py"
ruta_catalogo = raiz_proyecto / "Ingenieria_de_datos" / "Modificacion_muestra_1.py"

# Ejecutamos los tratamientos sin mostrar sus impresiones intermedias.
with redirect_stdout(StringIO()):
    datos_m3 = runpy.run_path(str(ruta_muestra))
    datos_catalogo = runpy.run_path(str(ruta_catalogo))

muestra_3_homologada = datos_m3["muestra_3_homologada"].copy(deep=True)

# Reutilizamos el catálogo que ya obtiene el tratamiento de la Muestra 1.
# No utilizamos ninguna variable ni columna de la Muestra 1 para comparar.
if "catalogo" in datos_catalogo:
    catalogo_homologado = datos_catalogo["catalogo"].copy(deep=True)
elif "catalogo_homologado" in datos_catalogo:
    catalogo_homologado = datos_catalogo["catalogo_homologado"].copy(deep=True)
else:
    raise KeyError(
        "Modificacion_muestra_1.py debe exponer 'catalogo' "
        "o 'catalogo_homologado'."
    )

# Trabajamos con un índice posicional único para factorizar y buscar candidatos.
muestra_3_homologada = muestra_3_homologada.reset_index(drop=True)
catalogo_homologado = catalogo_homologado.reset_index(drop=True)


# ==============================================================================
# 2. IDENTIFICACIÓN Y PREPARACIÓN DE LAS COLUMNAS DEL CATÁLOGO
# ==============================================================================

def normalizar_nombre(nombre):
    nombre = unicodedata.normalize("NFKD", str(nombre))
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    return " ".join(nombre.upper().split())


def buscar_columna(df, *alternativas):
    disponibles = {normalizar_nombre(c): c for c in df.columns}
    for alternativa in alternativas:
        encontrada = disponibles.get(normalizar_nombre(alternativa))
        if encontrada is not None:
            return encontrada
    raise KeyError(
        f"No se encontró ninguna de estas columnas: {alternativas}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


col_id = buscar_columna(catalogo_homologado, "ID")
col_marca = buscar_columna(catalogo_homologado, "MARCA")
col_tipo = buscar_columna(catalogo_homologado, "TIPO")
col_version = buscar_columna(catalogo_homologado, "VERSION")
col_modelo = buscar_columna(catalogo_homologado, "MODELO")
col_transmision = buscar_columna(catalogo_homologado, "TRANSMISIÓN", "TRANSMISION")
col_ocupantes = buscar_columna(
    catalogo_homologado,
    "OCUPANTES", "OCUPANTE", "NÚMERO DE OCUPANTES", "NUMERO DE OCUPANTES"
)


def a_ocupantes_enteros(serie):
    """Convierte 05, 5, 5.0 o '5 OCUP.' a 5, sin inventar valores."""
    texto = serie.astype("string").str.strip()
    extraidos = texto.str.extract(r"^(\d+)(?:\.0+)?(?:\s*OCUP\.?)?$", flags=re.I)[0]
    return pd.to_numeric(extraidos, errors="coerce").astype("Int64")


muestra_3_homologada["Ocupantes"] = a_ocupantes_enteros(
    muestra_3_homologada["Ocupantes"]
)
catalogo_homologado[col_ocupantes] = a_ocupantes_enteros(
    catalogo_homologado[col_ocupantes]
)

muestra_3_homologada["Modelo"] = pd.to_numeric(
    muestra_3_homologada["Modelo"], errors="coerce"
).astype("Int64")
catalogo_homologado[col_modelo] = pd.to_numeric(
    catalogo_homologado[col_modelo], errors="coerce"
).astype("Int64")

muestra_3_homologada["Transmisión"] = (
    muestra_3_homologada["Transmisión"].astype("string").str.strip().str.upper()
)
catalogo_homologado[col_transmision] = (
    catalogo_homologado[col_transmision].astype("string").str.strip().str.upper()
)

# La columna descripcion ORIGINAL de la Muestra 3 permanece intacta:
# no quitamos AUT, STD, ni ninguna otra palabra.

# Como en el algoritmo de la Muestra 1, A/S no agrega texto específico.
transmision_texto = (
    catalogo_homologado[col_transmision]
    .map({"A": "AUT", "S": "STD"})
    .fillna("")
)

partes_descripcion = pd.DataFrame({
    "MARCA": catalogo_homologado[col_marca],
    "TIPO": catalogo_homologado[col_tipo],
    "VERSION": catalogo_homologado[col_version],
    "TRANSMISIÓN": transmision_texto
})

catalogo_homologado["descripcion_generada"] = (
    partes_descripcion.fillna("").astype(str).agg(" ".join, axis=1)
    .str.split().str.join(" ").str.upper()
)


# ==============================================================================
# 3. GOWER CATEGÓRICO: MODELO, TRANSMISIÓN Y OCUPANTES (PESOS IGUALES)
# ==============================================================================

variables_gower = {
    "Modelo": col_modelo,
    "Transmisión": col_transmision,
    "Ocupantes": col_ocupantes,
}

n_muestra = len(muestra_3_homologada)
n_catalogo = len(catalogo_homologado)
n_variables = len(variables_gower)
codigos_muestra = np.empty((n_muestra, n_variables), dtype=np.int64)
codigos_catalogo = np.empty((n_catalogo, n_variables), dtype=np.int64)

for k, (col_muestra, col_cat) in enumerate(variables_gower.items()):
    valores = pd.concat(
        [muestra_3_homologada[col_muestra], catalogo_homologado[col_cat]],
        ignore_index=True
    ).astype("string")
    codigos, _ = pd.factorize(valores)
    codigos_muestra[:, k] = codigos[:n_muestra]
    codigos_catalogo[:, k] = codigos[n_muestra:]


def calcular_gower(indice_muestra, candidatos):
    vehiculo = codigos_muestra[indice_muestra]
    codigos_candidatos = codigos_catalogo[candidatos.index.to_numpy()]

    disponibles = (vehiculo != -1) & (codigos_candidatos != -1)
    coincidencias = ((codigos_candidatos == vehiculo) & disponibles).sum(axis=1)
    denominador = disponibles.sum(axis=1)

    return np.divide(
        coincidencias,
        denominador,
        out=np.full(len(candidatos), np.nan, dtype=float),
        where=denominador > 0
    )


# ==============================================================================
# 4. SIMILITUD TEXTUAL: TOKEN SET (75 %) + TOKEN SORT (25 %)
# ==============================================================================

def similitud_token_set(texto_muestra, texto_catalogo):
    if pd.isna(texto_muestra) or pd.isna(texto_catalogo):
        return float("nan")
    a, b = str(texto_muestra).strip(), str(texto_catalogo).strip()
    if not a or not b:
        return float("nan")
    return fuzz.token_set_ratio(a, b) / 100


def similitud_textual(texto_muestra, texto_catalogo):
    if pd.isna(texto_muestra) or pd.isna(texto_catalogo):
        return float("nan")

    a, b = str(texto_muestra).strip(), str(texto_catalogo).strip()
    if not a or not b:
        return float("nan")

    set_ratio = fuzz.token_set_ratio(a, b) / 100
    sort_ratio = fuzz.token_sort_ratio(a, b) / 100
    return 0.75 * set_ratio + 0.25 * sort_ratio


# ==============================================================================
# 5. CLAVETEO POR OCUPANTES -> TOP 15 TEXTUAL -> GOWER + TEXTO
# ==============================================================================

def clavetear_vehiculo(indice_muestra):
    vehiculo = muestra_3_homologada.iloc[indice_muestra]

    # Primer filtro OBLIGATORIO: mismo número de ocupantes.
    ocupantes = vehiculo["Ocupantes"]
    if pd.isna(ocupantes):
        return pd.DataFrame(), None

    catalogo_ocupantes = catalogo_homologado.loc[
        catalogo_homologado[col_ocupantes].eq(ocupantes).fillna(False)
    ]
    if catalogo_ocupantes.empty:
        return pd.DataFrame(), None

    # Preselección de las 15 DESCRIPCIONES DISTINTAS con mayor Token Set.
    descripciones = catalogo_ocupantes[["descripcion_generada"]].drop_duplicates().copy()
    descripciones["token_set_preseleccion"] = descripciones[
        "descripcion_generada"
    ].apply(lambda t: similitud_token_set(vehiculo["descripcion"], t))

    mejores_descripciones = descripciones.nlargest(
        15, "token_set_preseleccion"
    )
    if mejores_descripciones.empty:
        return pd.DataFrame(), None

    candidatos = catalogo_ocupantes.loc[
        catalogo_ocupantes["descripcion_generada"].isin(
            mejores_descripciones["descripcion_generada"]
        )
    ].copy()

    # Puntuación textual COMPLETA sobre los candidatos seleccionados:
    # 75 % Token Set + 25 % Token Sort.
    puntajes_texto = {
        t: similitud_textual(vehiculo["descripcion"], t)
        for t in mejores_descripciones["descripcion_generada"]
    }
    candidatos["similitud_descripcion"] = (
        candidatos["descripcion_generada"].map(puntajes_texto)
    )

    candidatos["similitud_gower"] = calcular_gower(indice_muestra, candidatos)

    # Similitud global: 50 % descripción y 50 % Gower.
    candidatos["similitud_final"] = (
        0.50 * candidatos["similitud_descripcion"]
        + 0.50 * candidatos["similitud_gower"]
    )

    # Si varias filas comparten descripción, gana la de mayor similitud final.
    candidatos = (
        candidatos.sort_values("similitud_final", ascending=False, kind="stable")
        .drop_duplicates(subset="descripcion_generada")
        .copy()
    )
    if candidatos.empty or candidatos["similitud_final"].isna().all():
        return candidatos, None

    mejor_candidato = candidatos.iloc[0]
    return candidatos, mejor_candidato


# ==============================================================================
# 6. ASIGNACIÓN DE ID, VALIDACIÓN Y EXPORTACIÓN
# ==============================================================================

umbral_similitud = 0.80
muestra_3_claveteada = muestra_3_homologada.copy(deep=True)
muestra_3_claveteada["ID_asignado"] = pd.Series(pd.NA, index=muestra_3_claveteada.index, dtype="object")

for indice_muestra in range(len(muestra_3_claveteada)):
    candidatos, mejor_candidato = clavetear_vehiculo(indice_muestra)
    if mejor_candidato is None:
        continue

    similitud_obtenida = mejor_candidato["similitud_final"]
    if pd.notna(similitud_obtenida) and similitud_obtenida > umbral_similitud:
        muestra_3_claveteada.at[indice_muestra, "ID_asignado"] = mejor_candidato[col_id]

carpeta_resultados = raiz_proyecto / "Resultados"
carpeta_resultados.mkdir(parents=True, exist_ok=True)
ruta_salida = carpeta_resultados / "Muestra_3_claveteada.xlsx"
muestra_3_claveteada.to_excel(ruta_salida, index=False)

asignados = int(muestra_3_claveteada["ID_asignado"].notna().sum())
print("\nClaveteo de la Muestra 3 finalizado.")
print("Total de vehículos:", len(muestra_3_claveteada))
print("Vehículos con clave asignada:", asignados)
print("Vehículos sin clave asignada:", len(muestra_3_claveteada) - asignados)
print("Archivo generado:", ruta_salida)

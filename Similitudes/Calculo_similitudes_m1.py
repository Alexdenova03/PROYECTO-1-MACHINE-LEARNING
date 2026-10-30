import runpy
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
import numpy as np
from rapidfuzz import fuzz


# Ejecutamos la ingeniería de datos sin mostrar sus impresiones.
ruta = (
    Path(__file__).resolve().parent.parent
    / "Ingenieria_de_datos"
    / "Modificacion_muestra_1.py"
)

with redirect_stdout(StringIO()):
    datos = runpy.run_path(str(ruta))

# Recuperamos la muestra homologada y una copia del catálogo.
muestra_1_homologada = datos["muestra_1_homologada"].copy()
catalogo_homologado = datos["catalogo"].copy()

# Homologamos las marcas adicionales indicadas para el claveteo.
# Trabajamos sobre una copia; la muestra y el catálogo originales no se modifican.
equivalencias_adicionales_marca = {
    "GMOTORS": "GMC",
    "GM MOTORS": "GMC",
    "GENERAL MOTORS": "GMC",
    "GM": "GMC",
    "MAYBACH": "MERCEDES BENZ"
}

muestra_1_homologada["Marca"] = (
    muestra_1_homologada["Marca"].replace(equivalencias_adicionales_marca)
)

# Unificamos las categorías de motor en mayúsculas.
catalogo_homologado["MOTOR"] = catalogo_homologado["MOTOR"].str.upper()


# Adaptamos la transmisión del catálogo al formato de las descripciones.
# Los registros A/S no reciben una transmisión específica.
transmision_texto = (
    catalogo_homologado["TRANSMISIÓN"]
    .map({"A": "AUT", "S": "STD"})
    .fillna("")
)

# Construimos la descripción del catálogo a partir de sus columnas.
# Conservamos todas las variables originales.
partes_descripcion = pd.DataFrame({
    "MARCA": catalogo_homologado["MARCA"],
    "TIPO": catalogo_homologado["TIPO"],
    "VERSION": catalogo_homologado["VERSION"],
    "TRANSMISIÓN": transmision_texto
})

catalogo_homologado["descripcion_generada"] = (
    partes_descripcion
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.split()
    .str.join(" ")
)


# Definimos las variables categóricas para calcular Gower.
variables_gower = {
    "Modelo": "MODELO",
    "Marca": "MARCA",
    "transmision": "TRANSMISIÓN",
    "motor": "MOTOR",
    "AC": "AIRE ACONDICIONADO"
}

n_muestra = len(muestra_1_homologada)
n_catalogo = len(catalogo_homologado)
n_variables = len(variables_gower)

codigos_muestra = np.empty((n_muestra, n_variables), dtype=int)
codigos_catalogo = np.empty((n_catalogo, n_variables), dtype=int)

# Factorizamos conjuntamente las categorías de ambas tablas.
# Las categorías coincidentes reciben el mismo código.
# Los valores faltantes se identifican mediante -1.
for k, (col_muestra, col_catalogo) in enumerate(variables_gower.items()):

    valores = pd.concat([
        muestra_1_homologada[col_muestra].reset_index(drop=True),
        catalogo_homologado[col_catalogo].reset_index(drop=True)
    ], ignore_index=True)

    codigos, _ = pd.factorize(valores)

    codigos_muestra[:, k] = codigos[:n_muestra]
    codigos_catalogo[:, k] = codigos[n_muestra:]


# Calculamos la similitud textual utilizando Token Set Ratio (75 %)
# y Token Sort Ratio (25 %).
def similitud_textual(texto_muestra, texto_catalogo):

    # Ignoramos las comparaciones con valores faltantes.
    if pd.isna(texto_muestra) or pd.isna(texto_catalogo):
        return float("nan")

    texto_muestra = str(texto_muestra).strip()
    texto_catalogo = str(texto_catalogo).strip()

    # Excluimos las cadenas vacías.
    if not texto_muestra or not texto_catalogo:
        return float("nan")

    similitud_set = fuzz.token_set_ratio(
        texto_muestra, texto_catalogo
    ) / 100

    similitud_sort = fuzz.token_sort_ratio(
        texto_muestra, texto_catalogo
    ) / 100

    return 0.75 * similitud_set + 0.25 * similitud_sort


# Calculamos Gower de forma vectorizada sobre los candidatos.
# Las cinco variables categóricas tienen el mismo peso.
def calcular_gower(indice_muestra, candidatos):

    vehiculo = codigos_muestra[indice_muestra]

    # Recuperamos las posiciones de los candidatos en el catálogo.
    posiciones = catalogo_homologado.index.get_indexer(candidatos.index)
    codigos_candidatos = codigos_catalogo[posiciones]

    # Excluimos las variables faltantes de cada comparación.
    disponibles = (vehiculo != -1) & (codigos_candidatos != -1)

    coincidencias = (
        (codigos_candidatos == vehiculo) & disponibles
    ).sum(axis=1)

    denominador = disponibles.sum(axis=1)

    # Calculamos la similitud evitando divisiones entre cero.
    similitudes = np.divide(
        coincidencias,
        denominador,
        out=np.full(len(candidatos), np.nan),
        where=denominador > 0
    )

    return similitudes


# Establecemos las ponderaciones de la similitud global.
pesos = pd.Series({
    "similitud_gower": 0.50,
    "similitud_descripcion": 0.40,
    "similitud_tipo": 0.10
})


# Identificamos el mejor candidato para un vehículo de la muestra.
def clavetear_vehiculo(indice_muestra):

    vehiculo = muestra_1_homologada.iloc[indice_muestra]

    # Buscamos únicamente vehículos con la misma marca.
    catalogo_marca = catalogo_homologado.loc[
        catalogo_homologado["MARCA"] == vehiculo["Marca"]
    ]

    # Si la marca no existe, no asignamos ninguna clave.
    if catalogo_marca.empty:
        return pd.DataFrame(), None

    # Identificamos las descripciones distintas de esa marca.
    descripciones = (
        catalogo_marca[["descripcion_generada"]]
        .drop_duplicates()
        .copy()
    )

    # Calculamos la similitud de cada descripción con la de la muestra.
    descripciones["similitud_descripcion"] = (
        descripciones["descripcion_generada"].apply(
            lambda texto: similitud_textual(
                vehiculo["descripcion"], texto
            )
        )
    )

    # Seleccionamos las 15 descripciones con mayor similitud.
    mejores_descripciones = descripciones.nlargest(
        15, "similitud_descripcion"
    )

    # Recuperamos los registros del catálogo asociados a esas descripciones.
    candidatos = catalogo_marca.loc[
        catalogo_marca["descripcion_generada"].isin(
            mejores_descripciones["descripcion_generada"]
        )
    ].copy()

    # Incorporamos las similitudes de descripción ya calculadas.
    puntajes_descripcion = dict(zip(
        mejores_descripciones["descripcion_generada"],
        mejores_descripciones["similitud_descripcion"]
    ))

    candidatos["similitud_descripcion"] = (
        candidatos["descripcion_generada"].map(puntajes_descripcion)
    )

    # Calculamos Gower sobre los candidatos seleccionados.
    candidatos["similitud_gower"] = calcular_gower(
        indice_muestra, candidatos
    )

    # Si una descripción aparece en varias filas, conservamos
    # el registro con mayor similitud de Gower.
    candidatos = (
        candidatos
        .sort_values("similitud_gower", ascending=False, kind="stable")
        .drop_duplicates(subset="descripcion_generada")
        .copy()
    )

    # Calculamos la similitud textual de Tipo.
    candidatos["similitud_tipo"] = candidatos["TIPO"].apply(
        lambda tipo: similitud_textual(vehiculo["Tipo"], tipo)
    )

    # Calculamos la media ponderada de las tres similitudes.
    # Los componentes faltantes no participan en el denominador.
    componentes = list(pesos.index)

    numerador = (
        candidatos[componentes]
        .mul(pesos)
        .sum(axis=1)
    )

    denominador = (
        candidatos[componentes]
        .notna()
        .mul(pesos)
        .sum(axis=1)
    )

    candidatos["similitud_final"] = numerador.div(
        denominador.where(denominador > 0)
    )

    # Ordenamos los candidatos por similitud global.
    candidatos = candidatos.sort_values(
        "similitud_final",
        ascending=False,
        kind="stable"
    )

    # Recuperamos el candidato con mayor similitud.
    mejor_candidato = candidatos.iloc[0]

    return candidatos, mejor_candidato


# Establecemos el umbral mínimo de aceptación.
umbral_similitud = 0.80

# Creamos la columna para almacenar las claves asignadas.
muestra_1_homologada["ID_asignado"] = pd.Series(
    pd.NA,
    index=muestra_1_homologada.index,
    dtype="object"
)

# Aplicamos el algoritmo a todas las observaciones de la Muestra 1.
for indice_muestra in range(len(muestra_1_homologada)):

    # Seleccionamos los candidatos y recuperamos el mejor.
    candidatos, mejor_candidato = clavetear_vehiculo(indice_muestra)

    # Si no encontramos candidatos, dejamos el ID vacío.
    if mejor_candidato is None:
        continue

    similitud_obtenida = mejor_candidato["similitud_final"]

    # Asignamos la clave si la similitud es superior a 0.80.
    if pd.notna(similitud_obtenida) and similitud_obtenida > umbral_similitud:

        muestra_1_homologada.at[
            muestra_1_homologada.index[indice_muestra],
            "ID_asignado"
        ] = mejor_candidato["ID"]


# ==============================================================================
# FASE 2A: GOWER DE MARCA Y MODELO (±5 AÑOS) + SIMILITUD TEXTUAL DE TIPO
# FASE 2B: MARCA + TIPO TEXTUAL + MAYOR VALOR COMERCIAL
# ==============================================================================
# La primera fase termina arriba. Ninguna clave de esa fase se reemplaza.

# Parámetros ajustables de los rescates.
tolerancia_modelo = 5
umbral_tipo_fase2 = 0.80
umbral_tipo_fase3 = 0.85
peso_gower_fase2 = 0.50
peso_tipo_fase2 = 0.50

# Preparamos únicamente las columnas auxiliares que requieren las nuevas fases.
# No modificamos los valores originales del catálogo.
marcas_catalogo_rescate = (
    catalogo_homologado["MARCA"].astype("string").str.strip().str.upper()
)
tipos_catalogo_rescate = (
    catalogo_homologado["TIPO"].astype("string").str.strip().str.upper()
)
modelos_catalogo_rescate = pd.to_numeric(
    catalogo_homologado["MODELO"], errors="coerce"
)
precios_catalogo_rescate = pd.to_numeric(
    catalogo_homologado["COMERCIAL"].astype("string")
    .str.replace(r"[$,\s]", "", regex=True),
    errors="coerce"
)


def clavetear_rescate(indice_muestra):
    """Devuelve (candidato, fase) o (None, None) si no hay correspondencia."""
    vehiculo = muestra_1_homologada.iloc[indice_muestra]

    if pd.isna(vehiculo["Marca"]) or pd.isna(vehiculo["Tipo"]):
        return None, None

    marca = str(vehiculo["Marca"]).strip().upper()
    tipo = str(vehiculo["Tipo"]).strip().upper()
    if not marca or not tipo:
        return None, None

    # Ambas fases requieren la misma marca comercial homologada.
    candidatos = catalogo_homologado.loc[
        marcas_catalogo_rescate.eq(marca).fillna(False)
    ].copy()
    if candidatos.empty:
        return None, None

    # Evaluamos el Tipo con la función textual ya usada en la primera fase:
    # 75 % Token Set Ratio + 25 % Token Sort Ratio.
    # Reutilizamos el puntaje por Tipo distinto para evitar cálculos repetidos.
    tipos_unicos = tipos_catalogo_rescate.loc[candidatos.index].dropna().unique()
    puntajes_tipo = {
        tipo_catalogo: similitud_textual(tipo, tipo_catalogo)
        for tipo_catalogo in tipos_unicos
        if tipo_catalogo
    }
    candidatos["similitud_tipo_rescate"] = (
        tipos_catalogo_rescate.loc[candidatos.index].map(puntajes_tipo)
    )

    # --------------------------------------------------------------------------
    # FASE 2A. Marca exacta, año a no más de 5 años y Tipo suficientemente similar.
    # --------------------------------------------------------------------------
    modelo_muestra = pd.to_numeric(
        pd.Series([vehiculo["Modelo"]]), errors="coerce"
    ).iloc[0]

    if pd.notna(modelo_muestra):
        diferencias = (
            modelos_catalogo_rescate.loc[candidatos.index] - modelo_muestra
        ).abs()

        elegibles = candidatos.loc[
            diferencias.le(tolerancia_modelo).fillna(False)
            & candidatos["similitud_tipo_rescate"].ge(umbral_tipo_fase2)
        ].copy()

        if not elegibles.empty:
            elegibles["diferencia_modelo"] = diferencias.loc[elegibles.index]

            # Gower mixto: Marca categórica (1 por igualdad) y Modelo numérico.
            # El Modelo se normaliza con la tolerancia de 5 años.
            similitud_modelo = (
                1 - elegibles["diferencia_modelo"] / tolerancia_modelo
            ).clip(lower=0, upper=1)
            elegibles["gower_rescate"] = (1.0 + similitud_modelo) / 2

            # Entre los elegibles, gana la mayor similitud combinada.
            elegibles["similitud_rescate"] = (
                peso_gower_fase2 * elegibles["gower_rescate"]
                + peso_tipo_fase2 * elegibles["similitud_tipo_rescate"]
            )
            mejor = elegibles.sort_values(
                ["similitud_rescate", "similitud_tipo_rescate", "diferencia_modelo"],
                ascending=[False, False, True],
                kind="stable"
            ).iloc[0]
            return mejor, "2A"

    # --------------------------------------------------------------------------
    # FASE 2B. Si sigue sin clave: misma marca, Tipo ≥ 85 % y mayor valor comercial.
    # No exigimos coincidencia de año-modelo ni de versión.
    # --------------------------------------------------------------------------
    elegibles = candidatos.loc[
        candidatos["similitud_tipo_rescate"].ge(umbral_tipo_fase3)
    ].copy()
    if elegibles.empty:
        return None, None

    elegibles["precio_comercial"] = precios_catalogo_rescate.loc[elegibles.index]
    elegibles = elegibles.dropna(subset=["precio_comercial"])
    if elegibles.empty:
        return None, None

    # El valor comercial es el criterio principal; el Tipo rompe empates.
    mejor = elegibles.sort_values(
        ["precio_comercial", "similitud_tipo_rescate"],
        ascending=[False, False],
        kind="stable"
    ).iloc[0]
    return mejor, "2B"


# Ejecutamos los rescates solo sobre los registros sin clave en la primera fase.
asignados_primera_fase = int(muestra_1_homologada["ID_asignado"].notna().sum())
pendientes = np.flatnonzero(muestra_1_homologada["ID_asignado"].isna().to_numpy())

asignados_fase_2a = 0
asignados_fase_2b = 0

for indice_muestra in pendientes:
    candidato, fase = clavetear_rescate(indice_muestra)
    if candidato is None:
        continue

    muestra_1_homologada.at[
        muestra_1_homologada.index[indice_muestra], "ID_asignado"
    ] = candidato["ID"]

    if fase == "2A":
        asignados_fase_2a += 1
    elif fase == "2B":
        asignados_fase_2b += 1


# Exportamos el resultado dentro del repositorio.
carpeta_resultados = Path(__file__).resolve().parent.parent / "Resultados"
carpeta_resultados.mkdir(parents=True, exist_ok=True)
ruta_salida = carpeta_resultados / "Muestra_1_claveteada.xlsx"

# Se sobrescribe el mismo Excel en cada ejecución.
muestra_1_homologada.to_excel(ruta_salida, index=False)

# Resumen del claveteo.
asignados = int(muestra_1_homologada["ID_asignado"].notna().sum())
print("\nClaveteo de la Muestra 1 finalizado.")
print("Total de vehículos:", len(muestra_1_homologada))
print("Asignados en primera fase:", asignados_primera_fase)
print("Asignados en fase 2A (Gower + Tipo):", asignados_fase_2a)
print("Asignados en fase 2B (mayor COMERCIAL):", asignados_fase_2b)
print("Vehículos con clave asignada:", asignados)
print("Vehículos sin clave asignada:", len(muestra_1_homologada) - asignados)
print("Archivo generado:", ruta_salida)

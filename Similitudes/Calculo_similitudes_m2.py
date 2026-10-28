import runpy
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
import numpy as np
from rapidfuzz import fuzz


# Ejecutamos la ingeniería de datos de la Muestra 2 sin mostrar sus impresiones.
ruta = (
    Path(__file__).resolve().parent.parent
    / "Ingenieria_de_datos"
    / "Modificacion_muestra_2.py"
)

with redirect_stdout(StringIO()):
    datos = runpy.run_path(str(ruta))

# Recuperamos las tablas previamente homologadas.
muestra_2_homologada = datos["muestra_2_homologada"].copy()
catalogo_homologado = datos["catalogo_homologado"].copy()


# Adaptamos la transmisión del catálogo al formato de las descripciones.
# Los registros A/S no reciben una transmisión específica.
transmision_texto = (
    catalogo_homologado["TRANSMISIÓN"]
    .map({"A": "AUT", "S": "STD"})
    .fillna("")
)

# Construimos una descripción adicional utilizando las columnas del catálogo.
# Conservamos todas las variables originales para las demás comparaciones.
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


# Definimos las seis variables categóricas que participarán en Gower.
# Incorporamos Ocupantes, que no estaba disponible en la Muestra 1.
variables_gower = {
    "modelo": "MODELO",
    "Marca": "MARCA",
    "transmision": "TRANSMISIÓN",
    "motor": "MOTOR",
    "AC": "AIRE ACONDICIONADO",
    "Ocupantes": "OCUPANTES"
}

n_muestra = len(muestra_2_homologada)
n_catalogo = len(catalogo_homologado)
n_variables = len(variables_gower)

codigos_muestra = np.empty((n_muestra, n_variables), dtype=int)
codigos_catalogo = np.empty((n_catalogo, n_variables), dtype=int)

# Factorizamos conjuntamente las categorías de ambas tablas.
# Los valores coincidentes reciben el mismo código y los faltantes reciben -1.
for k, (col_muestra, col_catalogo) in enumerate(variables_gower.items()):

    valores = pd.concat([
        muestra_2_homologada[col_muestra].reset_index(drop=True),
        catalogo_homologado[col_catalogo].reset_index(drop=True)
    ], ignore_index=True)

    codigos, _ = pd.factorize(valores)

    codigos_muestra[:, k] = codigos[:n_muestra]
    codigos_catalogo[:, k] = codigos[n_muestra:]


# Calculamos la similitud textual mediante Token Set Ratio (75 %)
# y Token Sort Ratio (25 %).
def similitud_textual(texto_muestra, texto_catalogo):

    # Ignoramos los valores faltantes.
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


# Calculamos Gower de forma vectorizada para un conjunto de candidatos.
# Las seis variables son categóricas y tienen el mismo peso.
def calcular_gower(indice_muestra, candidatos):

    vehiculo = codigos_muestra[indice_muestra]

    # Identificamos las posiciones de los candidatos en el catálogo.
    posiciones = catalogo_homologado.index.get_indexer(candidatos.index)
    codigos_candidatos = codigos_catalogo[posiciones]

    # Las variables faltantes no participan en la comparación.
    disponibles = (vehiculo != -1) & (codigos_candidatos != -1)

    coincidencias = (
        (codigos_candidatos == vehiculo) & disponibles
    ).sum(axis=1)

    denominador = disponibles.sum(axis=1)

    # Calculamos la media de las coincidencias disponibles.
    # Evitamos divisiones entre cero.
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


# Seleccionamos los candidatos y calculamos las similitudes de un vehículo.
def clavetear_vehiculo(indice_muestra):

    vehiculo = muestra_2_homologada.iloc[indice_muestra]

    # Filtramos el catálogo utilizando la marca homologada.
    catalogo_marca = catalogo_homologado.loc[
        catalogo_homologado["MARCA"] == vehiculo["Marca"]
    ]

    # Si la marca no existe en el catálogo, no asignamos ninguna clave.
    if catalogo_marca.empty:
        return pd.DataFrame(), None

    # Recuperamos las descripciones distintas disponibles para esa marca.
    descripciones = (
        catalogo_marca[["descripcion_generada"]]
        .drop_duplicates()
        .copy()
    )

    # Calculamos la similitud textual con cada descripción distinta.
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

    # Recuperamos los registros originales correspondientes a esas descripciones.
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

    # Calculamos Gower para los candidatos seleccionados.
    candidatos["similitud_gower"] = calcular_gower(
        indice_muestra, candidatos
    )

    # Si una descripción aparece en varios registros, conservamos
    # el que tenga mayor similitud de Gower.
    candidatos = (
        candidatos
        .sort_values("similitud_gower", ascending=False, kind="stable")
        .drop_duplicates(subset="descripcion_generada")
        .copy()
    )

    # Calculamos la similitud textual entre los tipos.
    candidatos["similitud_tipo"] = candidatos["TIPO"].apply(
        lambda tipo: similitud_textual(vehiculo["Tipo"], tipo)
    )

    # Calculamos la similitud global ponderada.
    # Las componentes faltantes se excluyen de la suma de pesos.
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

    # Ordenamos los candidatos según su similitud global.
    candidatos = candidatos.sort_values(
        "similitud_final",
        ascending=False,
        kind="stable"
    )

    # Recuperamos el candidato con mayor similitud.
    mejor_candidato = candidatos.iloc[0]

    return candidatos, mejor_candidato


# Establecemos el umbral mínimo para aceptar una correspondencia.
umbral_similitud = 0.80

# Creamos la columna donde se almacenarán las claves asignadas.
muestra_2_homologada["ID_asignado"] = pd.Series(
    pd.NA,
    index=muestra_2_homologada.index,
    dtype="object"
)

# Aplicamos el algoritmo a todas las observaciones de la Muestra 2.
for indice_muestra in range(len(muestra_2_homologada)):

    # Identificamos el candidato con mayor similitud.
    candidatos, mejor_candidato = clavetear_vehiculo(indice_muestra)

    # Si no existen candidatos, conservamos la clave vacía.
    if mejor_candidato is None:
        continue

    similitud_obtenida = mejor_candidato["similitud_final"]

    # Asignamos el ID solo cuando la similitud supera 0.80.
    if pd.notna(similitud_obtenida) and similitud_obtenida > umbral_similitud:

        muestra_2_homologada.at[
            muestra_2_homologada.index[indice_muestra],
            "ID_asignado"
        ] = mejor_candidato["ID"]


# Creamos la carpeta de resultados si todavía no existe.
carpeta_resultados = Path(__file__).resolve().parent.parent / "Resultados"
carpeta_resultados.mkdir(parents=True, exist_ok=True)

# Definimos la ruta del archivo de resultados de la Muestra 2.
ruta_salida = carpeta_resultados / "Muestra_2_claveteada.xlsx"

# Exportamos la muestra homologada con la columna de claves asignadas.
# Cada ejecución sobrescribe el archivo anterior sin crear duplicados.
muestra_2_homologada.to_excel(ruta_salida, index=False)

# Presentamos el resumen final del claveteo.
asignados = muestra_2_homologada["ID_asignado"].notna().sum()
sin_asignar = len(muestra_2_homologada) - asignados

print("\nClaveteo de la Muestra 2 finalizado.")
print("Total de vehículos:", len(muestra_2_homologada))
print("Vehículos con clave asignada:", asignados)
print("Vehículos sin clave asignada:", sin_asignar)
print("Archivo generado:", ruta_salida)

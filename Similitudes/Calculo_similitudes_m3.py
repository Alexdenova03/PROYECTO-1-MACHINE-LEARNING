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

# ==============================================================================
# 7. FASE 2: RESCATE DE LOS VEHÍCULOS SIN CLAVE
#    2A: MARCA + MODELO (±5 AÑOS) Y SIMILITUD TEXTUAL DE TIPO
#    2B: MARCA + MODELO EXACTO + TIPO TEXTUAL + MAYOR VALOR COMERCIAL
# ==============================================================================
# Se conserva por completo la fase 1 anterior. Estos rescates solo trabajan
# con las filas cuyo ID_asignado sigue siendo NA.

asignados_fase_1 = int(muestra_3_claveteada["ID_asignado"].notna().sum())
tolerancia_modelo = 5
umbral_tipo_fase_2a = 0.80

# NOTA: Para que 2B pueda recuperar vehículos no aceptados en 2A, el umbral
# textual debe ser inferior a 0.80, puesto que 2B exige ADEMÁS el año exacto.
umbral_tipo_fase_2b = 0.75
peso_gower_rescate = 0.50
peso_tipo_rescate = 0.50

col_comercial = buscar_columna(catalogo_homologado, "COMERCIAL")


def normalizar_texto(texto):
    """Unifica mayúsculas, acentos, puntuación y espacios para comparar."""
    if pd.isna(texto):
        return ""
    sin_acentos = unicodedata.normalize("NFKD", str(texto).upper())
    sin_acentos = "".join(
        caracter for caracter in sin_acentos
        if not unicodedata.combining(caracter)
    )
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", sin_acentos).split())


# Homologaciones SOLO para los rescates; no modifican la primera fase.
equivalencias_marcas = {
    "MBENZ": "MERCEDES BENZ",
    "MERCEDES BENZ": "MERCEDES BENZ",
    "MAYBACH": "MERCEDES BENZ",
    "LROVER": "LAND ROVER",
    "GENERAL MOTORS": "GMC",
    "GMOTORS": "GMC",
    "GM MOTORS": "GMC",
    "GM": "GMC",
}
equivalencias_marcas = {
    normalizar_texto(origen): normalizar_texto(destino)
    for origen, destino in equivalencias_marcas.items()
}

# Preparamos los campos auxiliares del catálogo, sin alterar sus valores.
marcas_catalogo_rescate = catalogo_homologado[col_marca].map(normalizar_texto)
tipos_catalogo_rescate = catalogo_homologado[col_tipo].map(normalizar_texto)
modelos_catalogo_rescate = pd.to_numeric(
    catalogo_homologado[col_modelo], errors="coerce"
)
precios_catalogo_rescate = pd.to_numeric(
    catalogo_homologado[col_comercial].astype("string")
    .str.replace(r"[$,\s]", "", regex=True),
    errors="coerce"
)

# Asociamos las marcas disponibles con sus tipos distintos.
indices_por_marca = {
    marca: np.flatnonzero(marcas_catalogo_rescate.eq(marca).to_numpy())
    for marca in marcas_catalogo_rescate.dropna().unique()
    if marca
}
tipos_por_marca = {
    marca: sorted(
        set(tipos_catalogo_rescate.iloc[indices]) - {""},
        key=lambda t: (len(t.split()), len(t)),
        reverse=True,
    )
    for marca, indices in indices_por_marca.items()
}

# Para marcas compuestas (LAND ROVER, MERCEDES BENZ) se toma el prefijo MÁS
# LARGO. También se tolera una palabra introductoria antes de la marca.
marcas_reconocidas = {
    **{marca: marca for marca in indices_por_marca},
    **equivalencias_marcas,
}
patrones_marcas = sorted(
    ((marca.split(), equivalente) for marca, equivalente in marcas_reconocidas.items()),
    key=lambda par: (len(par[0]), sum(len(x) for x in par[0])),
    reverse=True,
)


def extraer_marca_tipo(descripcion):
    """Devuelve (marca homologada, tipo inicial), o (None, None)."""
    palabras = normalizar_texto(descripcion).split()
    if not palabras:
        return None, None

    for inicio in (0, 1):
        for palabras_marca, marca in patrones_marcas:
            largo = len(palabras_marca)
            if palabras[inicio:inicio + largo] != palabras_marca:
                continue
            if marca not in indices_por_marca:
                continue

            resto = palabras[inicio + largo:]
            if not resto:
                return marca, None
            texto_restante = " ".join(resto)

            # El TIPO puede contener varias palabras: RANGE ROVER SPORT,
            # TOWN COUNTRY, SERIE 3, etc. Elegimos el prefijo más largo del
            # vocabulario de TIPO de esa misma marca.
            for tipo_catalogo in tipos_por_marca[marca]:
                opciones = [tipo_catalogo]
                # Hay TIPOS del catálogo que repiten la marca al inicio.
                if tipo_catalogo.startswith(marca + " "):
                    opciones.append(tipo_catalogo[len(marca) + 1:])
                for opcion in opciones:
                    if texto_restante == opcion or texto_restante.startswith(opcion + " "):
                        return marca, opcion

            # Si no hay TIPO literal, recuperamos los primeros tres tokens
            # sin inventar una coincidencia; el texto será evaluado con
            # Token Set y Token Sort contra los TIPOS reales del catálogo.
            return marca, " ".join(resto[:3])

    return None, None


# Extraemos Marca y Tipo solo de los vehículos que quedaron sin ID.
pendientes = np.flatnonzero(muestra_3_claveteada["ID_asignado"].isna().to_numpy())
marcas_tipos_pendientes = {
    indice: extraer_marca_tipo(muestra_3_claveteada.iloc[indice]["descripcion"])
    for indice in pendientes
}


def clavetear_rescate(indice_muestra):
    """Devuelve (ID, fase) para un pendiente, o (None, None)."""
    marca, tipo = marcas_tipos_pendientes[indice_muestra]
    if not marca or not tipo:
        return None, None

    indices = indices_por_marca.get(marca)
    if indices is None or len(indices) == 0:
        return None, None

    # Calculamos similitud textual una vez por cada Tipo distinto de la marca.
    similitudes_por_tipo = {
        t: similitud_textual(tipo, t)
        for t in tipos_por_marca[marca]
    }
    candidatos = catalogo_homologado.iloc[indices].copy()
    candidatos["similitud_tipo_rescate"] = (
        tipos_catalogo_rescate.iloc[indices].map(similitudes_por_tipo).to_numpy()
    )

    modelo = pd.to_numeric(
        pd.Series([muestra_3_claveteada.iloc[indice_muestra]["Modelo"]]),
        errors="coerce"
    ).iloc[0]
    if pd.isna(modelo):
        return None, None

    diferencia_modelo = (modelos_catalogo_rescate.iloc[indices] - modelo).abs()
    candidatos["diferencia_modelo"] = diferencia_modelo.to_numpy()

    # --------------------------------------------------------------------------
    # FASE 2A: GOWER CON MARCA Y MODELO (±5 AÑOS) + SIMILITUD DE TIPO ≥ 80 %.
    # --------------------------------------------------------------------------
    elegibles = candidatos.loc[
        candidatos["diferencia_modelo"].le(tolerancia_modelo)
        & candidatos["similitud_tipo_rescate"].ge(umbral_tipo_fase_2a)
    ].copy()

    if not elegibles.empty:
        # La marca ya coincide, por eso aporta 1. Modelo se compara como
        # variable numérica con distancia normalizada por cinco años.
        similitud_modelo = (
            1 - elegibles["diferencia_modelo"] / tolerancia_modelo
        ).clip(0, 1)
        elegibles["gower_rescate"] = (1 + similitud_modelo) / 2
        elegibles["similitud_rescate"] = (
            peso_gower_rescate * elegibles["gower_rescate"]
            + peso_tipo_rescate * elegibles["similitud_tipo_rescate"]
        )
        mejor = elegibles.sort_values(
            ["similitud_rescate", "similitud_tipo_rescate", "diferencia_modelo"],
            ascending=[False, False, True], kind="stable"
        ).iloc[0]
        return mejor[col_id], "2A"

    # --------------------------------------------------------------------------
    # FASE 2B: MISMA MARCA, MISMO AÑO-MODELO Y TIPO SIMILAR ≥ 75 %.
    # Solo aquí escogemos el vehículo de mayor valor comercial.
    # --------------------------------------------------------------------------
    elegibles = candidatos.loc[
        candidatos["diferencia_modelo"].eq(0)
        & candidatos["similitud_tipo_rescate"].ge(umbral_tipo_fase_2b)
    ].copy()
    if elegibles.empty:
        return None, None

    elegibles["precio_comercial"] = precios_catalogo_rescate.loc[elegibles.index]
    elegibles = elegibles.dropna(subset=["precio_comercial"])
    if elegibles.empty:
        return None, None

    mejor = elegibles.sort_values(
        ["precio_comercial", "similitud_tipo_rescate"],
        ascending=[False, False], kind="stable"
    ).iloc[0]
    return mejor[col_id], "2B"


asignados_fase_2a = 0
asignados_fase_2b = 0
for indice in pendientes:
    clave, fase = clavetear_rescate(indice)
    if clave is None or pd.isna(clave):
        continue

    muestra_3_claveteada.at[indice, "ID_asignado"] = clave
    if fase == "2A":
        asignados_fase_2a += 1
    else:
        asignados_fase_2b += 1

carpeta_resultados = raiz_proyecto / "Resultados"
carpeta_resultados.mkdir(parents=True, exist_ok=True)
ruta_salida = carpeta_resultados / "Muestra_3_claveteada.xlsx"
muestra_3_claveteada.to_excel(ruta_salida, index=False)

asignados = int(muestra_3_claveteada["ID_asignado"].notna().sum())
print("\nClaveteo de la Muestra 3 finalizado.")
print("Total de vehículos:", len(muestra_3_claveteada))
print("Asignados en primera fase:", asignados_fase_1)
print("Asignados en fase 2A (Gower + Tipo):", asignados_fase_2a)
print("Asignados en fase 2B (mayor COMERCIAL):", asignados_fase_2b)
print("Vehículos con clave asignada:", asignados)
print("Vehículos sin clave asignada:", len(muestra_3_claveteada) - asignados)
print("Archivo generado:", ruta_salida)

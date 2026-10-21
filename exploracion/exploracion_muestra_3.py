# %%
import pandas as pd

print("=== ANALISIS DESCRIPTIVO DE LOS DATOS DE LA MUESTRA 3 ===")

# 0. Importamos la muestra 3
muestra_3 = pd.read_excel("data/Muestras a Clavetear.xlsx", sheet_name="Muestra_3")

# 1. Revisamos la estructura general de los datos
print("\n=== ESTRUCTURA GENERAL ===\n")
muestra_3.info() ## Nos indica el numero de columnas, filas, valores null, y tipos de datos 
print("\nDimensiones (filas, columnas):", muestra_3.shape)

# Visualizamos las primeras observaciones
print("\n=== PRIMERAS 10 OBSERVACIONES ===\n")
print(muestra_3.head(10).to_string()) ## Nos muestra las primeras filas

# 2. Conteo de valores faltantes y valores distintos
print("\n=== VALORES FALTANTES Y DISTINTOS ===\n")

## Construimos un nuevo DataFrame para observar la composición de los datos
resumen_basico = pd.DataFrame({
    "Tipo_dato_colum": muestra_3.dtypes.astype(str),
    "Tipos_dato_cel": [
        muestra_3[col].dropna().map(type).unique().tolist() for col in muestra_3.columns
    ],
    "Val_distintos": muestra_3.nunique(dropna=True),
    "Val_faltantes": muestra_3.isna().sum()
})
print(resumen_basico.to_string())  ## Imprime la tabla de diagnostico

# ==============================================================================
# %%

# ==============================================================================
print("\n=== VALORES OBSERVADOS EN 'Ocupantes' ===\n")
frec_ocup = muestra_3["Ocupantes"].value_counts(dropna=False)  ## Cuenta cuantas veces se repite cada valor único
pct_ocup = (muestra_3["Ocupantes"].value_counts(dropna=False, normalize=True) * 100).round(2)  ## Obtenemos el porcentaje de cada categoría
print(pd.DataFrame({"Frecuencia": frec_ocup, "Porcentaje (%)": pct_ocup}).to_string())


print("\n=== VALORES OBSERVADOS EN 'Modelo' ===\n")
# Separamos valores numéricos de textos 

## Convertimos todos los elementos de la columna "Modelo" a números, en caso haber texto puro devuelve un NaN
modelo_num = pd.to_numeric(muestra_3["Modelo"], errors="coerce") 

## Identificamos cuales filas tenían algún texto o valor, además de encontrar el intento que fallo de convertir un número 
registros_modelo_texto = muestra_3[modelo_num.isna() & muestra_3["Modelo"].notna()]

# ==============================================================================
# %%

# ==============================================================================

## Contamos los valores que eran realmente números, y calculamos la proporción de datos válidos
print(f"Registros con año numérico en 'Modelo': {modelo_num.notna().sum()} ({modelo_num.notna().mean()*100:.2f}%)")

## Encontramos el año más antiguo d enuestra cartera y el más reciente (evitar errores de dedo)(valores atípicos)
print(f"Año mínimo numérico en 'Modelo': {modelo_num.min()}")
print(f"Año máximo numérico en 'Modelo': {modelo_num.max()}")

## Contamos cuantos años únicos diferentes existen, además de volver los valores enteros
print(f"Valores de año distintos en 'Modelo': {modelo_num.dropna().nunique()}")
print("Años observados en 'Modelo':", sorted(modelo_num.dropna().astype(int).unique().tolist()))

## Contamos los errores que pudimos capturas, debido a la válidación numérica
print(f"\nRegistros NO numéricos infiltrados en 'Modelo': {len(registros_modelo_texto)} ({len(registros_modelo_texto)/len(muestra_3)*100:.2f}%)")

## Extraemos las filas que resultaron ser texto erróneo de la columna "Modelo", además de contar las veces que se repite
print("Categorías infiltradas en 'Modelo':")
print(registros_modelo_texto["Modelo"].value_counts().to_string())

# ==============================================================================# %%
# %%
# ==============================================================================

print("\n=== CRUCE DE ANOMALÍAS (Ocupantes vs Modelo) ===")

## Buscamos celdas que no tengan la palabra OCUP en la columna 'Ocupantes', ó aquellas en donde hay texto infiltrado de la columna 'Modelo'
condicion_anomalia = ~muestra_3["Ocupantes"].astype(str).str.contains("OCUP", na=False) | modelo_num.isna()

## Contamos cuantas anomalías hay
print(f"Total de registros con desfase/anomalía: {condicion_anomalia.sum()} ({condicion_anomalia.mean()*100:.2f}%)")

## Filtramos las filas con anomalías, agrupamos las combinaciones de estas anomalías para ver si hay un patrón
print(
    muestra_3.loc[condicion_anomalia, ["Ocupantes", "Modelo"]]
    .value_counts(dropna=False)
    .to_string()
)

# Muestra de filas afectadas junto con su descripción
print("\nEjemplos de registros con columnas desplazadas:")
print(muestra_3.loc[condicion_anomalia].head(15).to_string())
# ==============================================================================

# %%
# ==============================================================================
print("\n=== TOP 10 DESCRIPCIONES MÁS FRECUENTES ===")

## Conteo de las veces que se repite cada texto único d ela columna 'Descripción'
print(muestra_3["descripcion"].value_counts().head(10).to_string())

# Muestra aleatoria de 40 descripciones distintas excluyendo el top 10

## Guardamos las 10 descripciones más frecuentes
frecuentes = muestra_3["descripcion"].value_counts().head(10).index

## Ubicamos las descripciones totalmente distintas
descripciones_restantes = muestra_3.loc[
    ~muestra_3["descripcion"].isin(frecuentes), "descripcion"
].drop_duplicates()

print("\n=== MUESTRA ALEATORIA DE 40 DESCRIPCIONES DISTINTAS ===")

## Seleccionamos una muestra aleatoria d elas 40 descripciones restantes
n_sample = min(40, len(descripciones_restantes))
print(descripciones_restantes.sample(n=n_sample, random_state=42).to_string(index=False))

# Extracción de la primera palabra (prefijo / marca) en 'descripcion'
## Convertimos la descripción en una lista de palabras separadas por espacios.
inicios = muestra_3["descripcion"].astype(str).str.split().str[0]

# ==============================================================================

# %%
# ==============================================================================

print(f"\n=== PREFIJOS INICIALES EN 'descripcion' ({inicios.nunique()} distintos) ===")

## Construimos una tabla de frecuencias con respecto a la primera palabra de cada descripción
frec_inicios = inicios.value_counts()
pct_inicios = (inicios.value_counts(normalize=True) * 100).round(2)
print(pd.DataFrame({"Frecuencia": frec_inicios, "Porcentaje (%)": pct_inicios}).to_string())

# Extracción de las primeras 2 palabras para detectar marcas compuestas (ej. ALFA ROMEO, LAND ROVER)
## Contamos las veces que se repite cada combinación de dos palabras
inicios_2 = muestra_3["descripcion"].astype(str).str.split().str[:2].str.join(" ")
print("\nTop 25 pares de palabras iniciales:")
print(inicios_2.value_counts().head(25).to_string())

# Verificación de terminaciones (Transmisión u otros patrones al final de la cadena)
## Obtenemos la última palabra de cada descripción e imprimimos las ´rimeras 20 más comunes
ultimas_palabras = muestra_3["descripcion"].astype(str).str.split().str[-1]

print(f"\n=== TERMINACIONES (ÚLTIMA PALABRA) EN 'descripcion' (Top 20 de {ultimas_palabras.nunique()}) ===")
## Evaluamos si el texto cumple con un patrón, aceptando puntos finales, las transmisiones más comunes
print(ultimas_palabras.value_counts().head(20).to_string())
transmision_final = muestra_3["descripcion"].astype(str).str.contains(
    r"\b(?:AUT|STD|MAN|CVT|TIPTRONIC|PDK|STRONIC|S-TRONIC|DCT|DSG)\.?\s*$", regex=True, na=False
)

## Contamos los registros que cumplen con la estructura estandarizada, buscamosa quellas que rompen con el estandar
print(f"\nDescripciones que terminan con indicador de transmisión: {transmision_final.sum()} ({transmision_final.mean()*100:.2f}%)")
print("Ejemplos que NO terminan con AUT/STD estándar:")
print(muestra_3.loc[~transmision_final, "descripcion"].drop_duplicates().head(20).to_string(index=False))

# ==============================================================================

# %%
# ==============================================================================
# ==============================================================================
# ANALISIS ESTADISTICO UNIVARIADO DE LA MUESTRA 3 (SIN MODIFICAR DATOS ORIGINALES)
# ==============================================================================
print("\n" + "="*70)
print("ANALISIS ESTADISTICO UNIVARIADO — MUESTRA 3")
print("="*70)

n_total = len(muestra_3)

# ------------------------------------------------------------------------------
# 1. VARIABLE DE TEXTO: 'descripcion'
# ------------------------------------------------------------------------------
frec_desc = muestra_3["descripcion"].value_counts()
n_distintas_desc = frec_desc.shape[0]
n_singletons = (frec_desc == 1).sum()
pct_singletons_distintas = (n_singletons / n_distintas_desc) * 100
pct_singletons_total = (n_singletons / n_total) * 100

n_repetidas = (frec_desc > 1).sum()
obs_en_repetidas = frec_desc[frec_desc > 1].sum()

print("\n--- 1. ANALISIS DE 'descripcion' ---")
print(f"Total de observaciones: {n_total}")
print(f"Descripciones distintas: {n_distintas_desc}")
print(f"Descripciones que aparecen una sola vez (singletons): {n_singletons} "
      f"({pct_singletons_distintas:.2f}% de las distintas | {pct_singletons_total:.2f}% del total de la muestra)")
print(f"Descripciones que se repiten (2 o más veces): {n_repetidas} "
      f"(concentran {obs_en_repetidas} observaciones, {obs_en_repetidas/n_total*100:.2f}% del total)")

# Distribución de frecuencias de repetición (cuántas aparecen 1 vez, 2 veces, etc.)
print("\nDistribución del número de repeticiones por descripción:")
dist_rep = frec_desc.value_counts().sort_index()
print(pd.DataFrame({
    "Num_Descripciones": dist_rep,
    "Observaciones_Acumuladas": dist_rep.index * dist_rep.values,
    "Porcentaje_Muestra (%)": (dist_rep.index * dist_rep.values / n_total * 100).round(2)
}).to_string())

# Top 15 descripciones más frecuentes con frecuencia absoluta y relativa
top15_desc = pd.DataFrame({
    "Frecuencia_Abs": frec_desc.head(15),
    "Frecuencia_Rel (%)": (frec_desc.head(15) / n_total * 100).round(2)
})
print("\nTop 15 descripciones más frecuentes:")
print(top15_desc.to_string())

# Estadísticos de longitud (caracteres y palabras) como diagnóstico de estructura textual
long_caracteres = muestra_3["descripcion"].str.len()
long_palabras = muestra_3["descripcion"].str.split().str.len()

stats_texto = pd.DataFrame({
    "Longitud_Caracteres": [
        long_caracteres.min(), long_caracteres.quantile(0.25), long_caracteres.median(),
        long_caracteres.mean(), long_caracteres.quantile(0.75), long_caracteres.max(),
        long_caracteres.std()
    ],
    "Numero_Palabras": [
        long_palabras.min(), long_palabras.quantile(0.25), long_palabras.median(),
        long_palabras.mean(), long_palabras.quantile(0.75), long_palabras.max(),
        long_palabras.std()
    ]
}, index=["Mínimo", "Q1 (25%)", "Mediana (50%)", "Media", "Q3 (75%)", "Máximo", "Desv. Estándar"]).round(2)

print("\nEstadísticos descriptivos de la longitud de 'descripcion':")
print(stats_texto.to_string())

# ------------------------------------------------------------------------------
# 2. VARIABLE 'Ocupantes' (FRECUENCIAS Y ESTADÍSTICOS SOBRE VÁLIDOS)
# ------------------------------------------------------------------------------
print("\n--- 2. ANALISIS DE 'Ocupantes' ---")
ocup_str = muestra_3["Ocupantes"].astype(str).str.strip()
es_ocup_valido = ocup_str.str.contains("OCUP", na=False)
n_ocup_validos = es_ocup_valido.sum()

# Tabla de categorías válidas vs registros desplazados
frec_ocup_val = ocup_str[es_ocup_valido].value_counts()
tabla_ocup_completa = pd.DataFrame({
    "Frecuencia_Abs": frec_ocup_val,
    "% sobre Total (N=803)": (frec_ocup_val / n_total * 100).round(2),
    "% sobre Válidos (n=736)": (frec_ocup_val / n_ocup_validos * 100).round(2)
})
print("Frecuencias de categorías válidas en 'Ocupantes':")
print(tabla_ocup_completa.to_string())
print(f"Registros sin información válida de ocupantes en la columna (desplazados): "
      f"{n_total - n_ocup_validos} ({(n_total - n_ocup_validos)/n_total*100:.2f}% del total)")

# Extracción temporal del número entero de ocupantes (solo sobre los 736 válidos) para estadísticos
ocup_num_validos = ocup_str[es_ocup_valido].str.extract(r"(\d+)")[0].astype(int)
stats_ocup = pd.Series({
    "n válido": len(ocup_num_validos),
    "Mínimo": ocup_num_validos.min(),
    "Q1 (25%)": ocup_num_validos.quantile(0.25),
    "Mediana (50%)": ocup_num_validos.median(),
    "Media": round(ocup_num_validos.mean(), 2),
    "Q3 (75%)": ocup_num_validos.quantile(0.75),
    "Máximo": ocup_num_validos.max(),
    "Desv. Estándar": round(ocup_num_validos.std(), 2),
    "Rango Intercuartílico (RIC)": ocup_num_validos.quantile(0.75) - ocup_num_validos.quantile(0.25)
})
print("\nEstadísticos descriptivos de 'Ocupantes' (sobre n=736 válidos):")
print(stats_ocup.to_string())

# ------------------------------------------------------------------------------
# 3. VARIABLE CUANTITATIVA DISCRETA: 'Modelo' (AÑO-MODELO)
# ------------------------------------------------------------------------------
print("\n--- 3. ANALISIS DE 'Modelo' ---")
modelo_num = pd.to_numeric(muestra_3["Modelo"], errors="coerce")
modelos_validos = modelo_num.dropna().astype(int)
n_mod_validos = len(modelos_validos)

stats_modelo = pd.Series({
    "n válido": n_mod_validos,
    "Faltantes implícitos (desplazados)": n_total - n_mod_validos,
    "% Faltantes implícitos": round((n_total - n_mod_validos) / n_total * 100, 2),
    "Mínimo": modelos_validos.min(),
    "Q1 (25%)": modelos_validos.quantile(0.25),
    "Mediana (50%)": modelos_validos.median(),
    "Media": round(modelos_validos.mean(), 2),
    "Q3 (75%)": modelos_validos.quantile(0.75),
    "Máximo": modelos_validos.max(),
    "Desv. Estándar": round(modelos_validos.std(), 2),
    "Rango Intercuartílico (RIC)": modelos_validos.quantile(0.75) - modelos_validos.quantile(0.25)
})
print("Estadísticos descriptivos de 'Modelo' (sobre n=736 años numéricos válidos):")
print(stats_modelo.to_string())

# Frecuencias por año-modelo (ordenado por frecuencia y cronológicamente)
frec_anio = modelos_validos.value_counts()
tabla_anios_top10 = pd.DataFrame({
    "Frecuencia_Abs": frec_anio.head(10),
    "% sobre Total (N=803)": (frec_anio.head(10) / n_total * 100).round(2),
    "% sobre Válidos (n=736)": (frec_anio.head(10) / n_mod_validos * 100).round(2)
})
print("\nTop 10 años-modelo más frecuentes:")
print(tabla_anios_top10.to_string())

# Distribución agrupada por lustros/décadas para analizar la concentración temporal
bins_anios = [1988, 1995, 2000, 2005, 2010, 2015, 2020]
labels_anios = ["1989-1995", "1996-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020"]
frec_periodos = pd.cut(modelos_validos, bins=bins_anios, labels=labels_anios).value_counts().sort_index()

tabla_periodos = pd.DataFrame({
    "Frecuencia_Abs": frec_periodos,
    "% sobre Total (N=803)": (frec_periodos / n_total * 100).round(2),
    "% sobre Válidos (n=736)": (frec_periodos / n_mod_validos * 100).round(2)
})
print("\nDistribución de 'Modelo' por periodos:")
print(tabla_periodos.to_string())

import pandas as pd

#ANALISIS DESCRIPTIVO DE LOS DATOS DE LA MUESTRA 1
# Importamos la primera muestra del archivo Excel.
muestra_1 = pd.read_excel("data\Muestras a Clavetear.xlsx", sheet_name="Muestra_1")

# Examinamos la estructura de los datos.
muestra_1.info()

# Visualizamos las primeras cinco observaciones.
print(muestra_1.head())

# Contamos los valores faltantes de cada variable.
print(muestra_1.isna().sum())

# Calculamos el porcentaje de valores faltantes.
print(muestra_1.isna().mean() * 100)

# Identificamos la cantidad de valores distintos de cada variable.
print(muestra_1.nunique())

# Identificamos los valores que toman las posibles variables binarias.
variables_binarias = ["AC", "Quemacocos", "TURBO", "hibrido"]

for variable in variables_binarias:
    print(f"{variable}: {muestra_1[variable].unique()}")

# Identificamos las categorías de las especificaciones mecánicas.
variables_mecanicas = ["motor", "transmision", "traccion"]

for variable in variables_mecanicas:
    print(f"{variable}: {muestra_1[variable].dropna().unique()}")

# Verificamos que los años-modelo estén dentro de un intervalo plausible.
anio_min = 1900
anio_max = pd.Timestamp.today().year + 1

modelos_invalidos = muestra_1[
    ~muestra_1["Modelo"].between(anio_min, anio_max)
]

print("Año mínimo observado:", muestra_1["Modelo"].min())
print("Año máximo observado:", muestra_1["Modelo"].max())
print("Registros fuera del intervalo:", len(modelos_invalidos))

print(modelos_invalidos[["Modelo", "descripcion"]])

# Identificamos las marcas registradas en la muestra.
print(sorted(muestra_1["Marca"].unique()))

# Identificamos los valores distintos de cilindrada, ordenados de menor a mayor.
print(sorted(muestra_1["cilindrada"].dropna().unique()))

# Examinamos las categorías más frecuentes y una selección aleatoria de tipos.
frecuencias = muestra_1["Tipo"].value_counts()

print(frecuencias.head(10))

tipos_restantes = frecuencias.index[10:]

print(pd.Series(tipos_restantes).sample(n=20, random_state=42).to_list())


# Examinamos las descripciones asociadas con las categorías ambiguas.
for tipo in ["CLASICO", "ALLBLACK"]:
    print(f"\nCategoría: {tipo}")

    registros = muestra_1.loc[
        muestra_1["Tipo"] == tipo,
        ["Tipo", "descripcion"]
    ].drop_duplicates()

    print(registros.to_string(index=False))


# Identificamos las 10 descripciones más frecuentes de la muestra.
print(muestra_1["descripcion"].value_counts().head(10).to_string())

# Excluimos las 10 descripciones más frecuentes de la muestra.
frecuentes = muestra_1["descripcion"].value_counts().head(10).index

descripciones = muestra_1.loc[
    ~muestra_1["descripcion"].isin(frecuentes),
    "descripcion"
].drop_duplicates()

# Seleccionamos aleatoriamente 40 descripciones distintas.
muestra_descripciones = descripciones.sample(n=40, random_state=42)

print(muestra_descripciones.to_string(index=False))

# Verificamos si las descripciones terminan indicando la transmisión.
transmision_final = muestra_1["descripcion"].str.contains(
    r"\b(?:AUT|STD)\.?\s*$", regex=True, na=False
)

print("Descripciones que cumplen:", transmision_final.sum())
print("Porcentaje:", transmision_final.mean() * 100)

# Identificamos las descripciones que no siguen este patrón.
print(muestra_1.loc[~transmision_final, "descripcion"].drop_duplicates().to_string(index=False))


# Extraemos la primera palabra de cada descripción y listamos los valores distintos.
inicios = muestra_1["descripcion"].str.split().str[0]

print(sorted(inicios.unique()))
print("Cantidad de inicios distintos:", inicios.nunique())


# Identificamos qué marcas están asociadas con cada prefijo de descripción.
comparacion = muestra_1.assign(inicio=inicios)

print(
    comparacion.groupby(["inicio", "Marca"])
    .size()
    .to_string()
)


# Extraemos la segunda o tercera palabra según la longitud del prefijo.
def extraer_tipo(descripcion):
    palabras = descripcion.split()
    posicion = 2 if len(palabras[0]) == 2 else 1

    return palabras[posicion] if len(palabras) > posicion else None

tipos_extraidos = muestra_1["descripcion"].apply(extraer_tipo)

# Comparamos las palabras extraídas con la variable Tipo.
coincidencias = (
    tipos_extraidos.str.upper()
    == muestra_1["Tipo"].astype(str).str.upper()
)

print("Coincidencias exactas:", coincidencias.sum())
print("Porcentaje de coincidencias:", coincidencias.mean() * 100)

# Examinamos algunas de las diferencias encontradas.
comparacion = pd.DataFrame({
    "descripcion": muestra_1["descripcion"],
    "Tipo": muestra_1["Tipo"],
    "Extraido": tipos_extraidos
})

print(comparacion.loc[~coincidencias].head(15).to_string(index=False))

#ANALISIS ESTADISTICO DE LA MUESTRA 1

# Calculamos las frecuencias absolutas y relativas de las variables binarias.
variables_binarias = ["AC", "Quemacocos", "TURBO", "hibrido"]

for variable in variables_binarias:
    frecuencias = muestra_1[variable].value_counts().sort_index()

    tabla = pd.DataFrame({
        "Frecuencia": frecuencias,
        "Porcentaje (%)": (frecuencias / len(muestra_1) * 100).round(2)
    })

    print(f"\n{variable}")
    print(tabla.to_string())




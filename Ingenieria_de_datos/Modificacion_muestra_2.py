
import pandas as pd

# Cargamos la Muestra 2 y el catálogo vehicular.
muestra_2 = pd.read_excel("data/Muestras a Clavetear.xlsx", sheet_name="Muestra_2")
catalogo = pd.read_excel("data/Catalogo Proyecto 1.xlsx")

# Creamos copias para realizar las homologaciones sin alterar los datos originales.
muestra_2_homologada = muestra_2.copy()
catalogo_homologado = catalogo.copy()


# Convertimos la capacidad de ocupantes al formato numérico del catálogo.
# Por ejemplo, "05 OCUP." se transforma en 5.
muestra_2_homologada["Ocupantes"] = (
    muestra_2_homologada["Ocupantes"]
    .str.extract(r"(\d+)\s*OCUP\.", expand=False)
    .astype(int)
)

print("Distribución de ocupantes homologados:")
print(muestra_2_homologada["Ocupantes"].value_counts().sort_index())


# Comparamos las marcas de la muestra con las del catálogo.
# Las marcas sin correspondencia se conservan sin modificaciones.
marcas_muestra = set(muestra_2_homologada["Marca"])
marcas_catalogo = set(catalogo["MARCA"])

marcas_coincidentes = marcas_muestra & marcas_catalogo
marcas_sin_correspondencia = marcas_muestra - marcas_catalogo

print("\nMarcas distintas de la muestra:", len(marcas_muestra))
print("Marcas coincidentes con el catálogo:", len(marcas_coincidentes))
print("Marcas sin correspondencia:", len(marcas_sin_correspondencia))
print("\nMarcas sin correspondencia:")
print(sorted(marcas_sin_correspondencia))


# Homologamos aire acondicionado con la nomenclatura del catálogo.
# SA = sin aire acondicionado y CA = con aire acondicionado.
muestra_2_homologada["AC"] = (
    muestra_2_homologada["AC"].replace({
        0: "SA",
        1: "CA"
    })
)

# Homologamos quemacocos con la nomenclatura de TECHO del catálogo.
# SQ = sin quemacocos y CQ = con quemacocos.
muestra_2_homologada["Quemacocos"] = (
    muestra_2_homologada["Quemacocos"].replace({
        0: "SQ",
        1: "CQ"
    })
)


# Eliminamos el punto final y los espacios sobrantes de las descripciones.
# Esto también corrige terminaciones como "AUT ." o "STD .".
muestra_2_homologada["descripcion"] = (
    muestra_2_homologada["descripcion"]
    .str.strip()
    .str.rstrip(".")
    .str.strip()
)

# Extraemos la transmisión indicada al final de cada descripción.
transmision_descripcion = (
    muestra_2_homologada["descripcion"].str.split().str[-1]
)

# Identificamos las discrepancias entre la columna transmision y la descripción.
diferencias = (
    transmision_descripcion.isin(["AUT", "STD"])
    & muestra_2_homologada["transmision"].ne(transmision_descripcion)
)

# Corregimos las discrepancias utilizando la transmisión de la descripción.
muestra_2_homologada.loc[diferencias, "transmision"] = (
    transmision_descripcion.loc[diferencias]
)

print("\nTransmisiones corregidas:", diferencias.sum())

# Verificamos que todas las descripciones terminen en AUT o STD.
print("Descripciones sin terminación válida:",
      (~transmision_descripcion.isin(["AUT", "STD"])).sum())

# Homologamos las transmisiones con la nomenclatura del catálogo.
# AUT = automática (A) y STD = estándar (S).
muestra_2_homologada["transmision"] = (
    muestra_2_homologada["transmision"].replace({
        "AUT": "A",
        "STD": "S"
    })
)

print("\nDistribución de transmisión homologada:")
print(muestra_2_homologada["transmision"].value_counts())


# Normalizamos las configuraciones de motor del catálogo en mayúsculas.
# Por ejemplo, "l4" se transforma en "L4", conservando los valores faltantes.
catalogo_homologado["MOTOR"] = (
    catalogo_homologado["MOTOR"].str.upper()
)

print("\nDistribución de motor en el catálogo homologado:")
print(catalogo_homologado["MOTOR"].value_counts(dropna=False))


# Comparamos las categorías de Tipo de la muestra con las del catálogo.
# Convertimos temporalmente los valores a texto para incluir los tipos numéricos.
# No modificamos esta variable; las diferencias se tratarán en la disimilitud.
tipos_muestra = set(
    muestra_2_homologada["Tipo"].dropna().astype(str).str.strip()
)

tipos_catalogo = set(
    catalogo_homologado["TIPO"].dropna().astype(str).str.strip()
)

tipos_iguales = tipos_muestra & tipos_catalogo
tipos_solo_muestra = tipos_muestra - tipos_catalogo
tipos_solo_catalogo = tipos_catalogo - tipos_muestra

print("\nTipos coincidentes:", len(tipos_iguales))
print("Tipos exclusivos de la muestra:", len(tipos_solo_muestra))
print("Tipos exclusivos del catálogo:", len(tipos_solo_catalogo))

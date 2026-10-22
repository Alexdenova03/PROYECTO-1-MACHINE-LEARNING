
import pandas as pd

# Cargamos la Muestra 1 y el catálogo vehicular.
muestra_1 = pd.read_excel("data/Muestras a Clavetear.xlsx", sheet_name="Muestra_1")
catalogo = pd.read_excel("data/Catalogo Proyecto 1.xlsx")

# Creamos una copia para realizar las transformaciones sin alterar los datos originales.
muestra_1_homologada = muestra_1.copy()


# Homologamos las abreviaturas de marca con las denominaciones utilizadas en el catálogo.
equivalencias_marca = {
    "MBENZ": "MERCEDES BENZ",
    "LROVER": "LAND ROVER"
}

muestra_1_homologada["Marca"] = (
    muestra_1_homologada["Marca"].replace(equivalencias_marca)
)


# Comprobamos las coincidencias entre las marcas homologadas y las del catálogo.
marcas_muestra = set(muestra_1_homologada["Marca"])
marcas_catalogo = set(catalogo["MARCA"])

marcas_coincidentes = marcas_muestra & marcas_catalogo
marcas_sin_correspondencia = marcas_muestra - marcas_catalogo

print("Marcas distintas de la muestra:", len(marcas_muestra))
print("Marcas coincidentes con el catálogo:", len(marcas_coincidentes))
print("Marcas sin correspondencia:", len(marcas_sin_correspondencia))
print("\nMarcas sin correspondencia:")
print(sorted(marcas_sin_correspondencia))


# Identificamos las marcas de la muestra que no existen en el catálogo.
# Estas denominaciones se conservan sin cambios.
marcas_sin_correspondencia = sorted(
    set(muestra_1_homologada["Marca"]) - set(catalogo["MARCA"])
)

print("Marcas sin correspondencia en el catálogo:")
print(marcas_sin_correspondencia)


# Homologamos aire acondicionado: 0 = sin aire y 1 = con aire.
muestra_1_homologada["AC"] = (
    muestra_1_homologada["AC"].replace({
        0: "SA",
        1: "CA"
    })
)


# Homologamos quemacocos con la nomenclatura de TECHO del catálogo.
# SQ = sin quemacocos y CQ = con quemacocos.
muestra_1_homologada["Quemacocos"] = (
    muestra_1_homologada["Quemacocos"].replace({
        0: "SQ",
        1: "CQ"
    })
)


# Eliminamos los espacios sobrantes y los puntos al final de las descripciones.
muestra_1_homologada["descripcion"] = (
    muestra_1_homologada["descripcion"].str.strip().str.rstrip(".")
)

# Extraemos la última palabra de cada descripción, que corresponde a AUT o STD.
transmision_descripcion = (
    muestra_1_homologada["descripcion"].str.split().str[-1]
)

# Identificamos las discrepancias entre la columna transmision y la descripción.
diferencias = (
    transmision_descripcion.isin(["AUT", "STD"])
    & muestra_1_homologada["transmision"].ne(transmision_descripcion)
)

# Corregimos las discrepancias utilizando la transmisión indicada en la descripción.
muestra_1_homologada.loc[diferencias, "transmision"] = (
    transmision_descripcion.loc[diferencias]
)

print("\nTransmisiones corregidas:", diferencias.sum())
print("\nDistribución de transmisión homologada:")
print(muestra_1_homologada["transmision"].value_counts())

# Comprobamos que todas las descripciones terminen en AUT o STD.
print("\nDescripciones sin terminación válida:",
      (~transmision_descripcion.isin(["AUT", "STD"])).sum())


# Homologamos la transmisión con la nomenclatura del catálogo.
# AUT = automática (A) y STD = estándar (S).
muestra_1_homologada["transmision"] = (
    muestra_1_homologada["transmision"].replace({
        "AUT": "A",
        "STD": "S"
    })
)


# Comparamos las denominaciones de Tipo de la muestra con las del catálogo.
# No modificamos esta variable; sus diferencias se tratarán durante la disimilitud.
tipos_muestra = set(muestra_1_homologada["Tipo"].dropna().astype(str).str.strip())
tipos_catalogo = set(catalogo["TIPO"].dropna().astype(str).str.strip())

tipos_iguales = tipos_muestra & tipos_catalogo
tipos_solo_muestra = tipos_muestra - tipos_catalogo
tipos_solo_catalogo = tipos_catalogo - tipos_muestra

print("\nTipos coincidentes:", len(tipos_iguales))
print("Tipos exclusivos de la muestra:", len(tipos_solo_muestra))
print("Tipos exclusivos del catálogo:", len(tipos_solo_catalogo))

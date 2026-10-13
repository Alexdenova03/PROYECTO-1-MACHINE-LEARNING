
# Proyecto I - Machine Learning

## Seminario de Aplicaciones Actuariales I
**Aprendizaje Automático y Business Intelligence en Seguro Automotriz**

Facultad de Ciencias, UNAM  
Semestre 2027-I

## Descripción

El proyecto consiste en construir y evaluar un homologador de descripciones vehiculares para una aseguradora ficticia.

El objetivo es realizar el proceso de claveteo, es decir, asignar a los vehículos de tres flotillas externas la clave correspondiente de un catálogo vehicular.

Para ello, se utilizarán técnicas de análisis exploratorio, ingeniería de datos y algoritmos de comparación de registros.

## Objetivos

1. Realizar el análisis exploratorio de las tres flotillas.
2. Construir un diccionario de datos para cada muestra.
3. Identificar valores ausentes, inconsistencias y problemas de formato.
4. Aplicar técnicas de limpieza y homologación.
5. Seleccionar algoritmos de comparación adecuados.
6. Evaluar la calidad de las correspondencias obtenidas.

## Estructura del repositorio

    Proyecto-I/
    │
    ├── datos/
    │
    ├── exploracion/
    │   ├── exploracion_muestra_1.py
    │   ├── exploracion_muestra_2.py
    │   └── exploracion_muestra_3.py
    │
    ├── diccionarios/
    │   ├── diccionario_muestra_1.ipynb
    │   ├── diccionario_muestra_2.ipynb
    │   └── diccionario_muestra_3.ipynb
    │
    ├── .gitignore
    └── README.md

## Metodología

### Etapa 1. Análisis exploratorio

Las tres flotillas se analizarán inicialmente de manera independiente.

Se estudiarán las variables disponibles, sus tipos, distribuciones, valores ausentes y posibles inconsistencias.

### Etapa 2. Ingeniería de datos

Se realizarán las transformaciones necesarias para facilitar la comparación entre las flotillas y el catálogo vehicular.

### Etapa 3. Aplicación de algoritmos

Se evaluarán diferentes técnicas de comparación y disimilitud para encontrar correspondencias entre los registros.

Se determinará si es adecuado implementar un algoritmo general o algoritmos específicos para cada flotilla.

### Etapa 4. Evaluación

Se evaluará la calidad de las asignaciones utilizando los registros de referencia proporcionados por el curso.

La meta establecida es alcanzar al menos un 80% de clasificación correcta.

## Herramientas

- Python
- Visual Studio Code
- Jupyter Notebook
- Git y GitHub

Las bibliotecas utilizadas se documentarán conforme avance el proyecto.

## Estado del proyecto

En desarrollo.

## Fecha de entrega

27 de octubre de 2026.

<h1 align="center">Proyecto Telecom</h1>

Modelo predictivo para pronosticar tasa de cancelación de la compañía Interconnect.

Este proyecto contiene un código para cargar, preprocesar datos y entrenar un modelo de machine learning. A continuación, se explica cómo configurarlo y ejecutarlo correctamente.

<h3 align="left">Requisitos</h3>
Antes de comenzar, asegúrate de tener los siguientes requisitos instalados:

 - Python 3.11.5
 - Bibliotecas Python necesarias: Puedes instalarlas utilizando el archivo requirements.txt.

<h3 align="left">Instalar Dependencias</h3>

 - Clona este repositorio en tu máquina local:

        git clone https://github.com/tu-usuario/tu-repositorio.git

 - Navega al directorio del proyecto:
        
        cd tu-repositorio
        
 - Instala las dependencias del proyecto utilizando pip:

        pip install -r requirements.txt

<h3 align="left">Estructura del Proyecto</h3>
    
 - imports.py: Archivo que contiene todas las importaciones de bibliotecas necesarias para el proyecto.
 - load_data.py: Módulo que contiene las funciones para cargar, fusionar y preprocesar los datos, así como la selección de características y etiquetas.
 - trainning.py: Módulo con las funciones de entrenamiento del modelo.

<h3 align="left">Ejecutar el Código</h3>
Para ejecutar el código, sigue estos pasos:

1. Asegúrate de tener los archivos de datos listos para cargar en el directorio correspondiente (si es necesario).
2. Ejecuta el script principal que llama a las funciones de cada archivo para cargar, preprocesar los datos y entrenar el modelo:

        python main.py

Este script importará las siguientes funciones desde los módulos correspondientes:

 - load_data(): Carga los datos desde las fuentes definidas.
 - merge_dataframes(): Fusiona los DataFrames necesarios.
 - preprocess_data(): Preprocesa los datos para preparar el modelo.
 - features_target(): Define las características y etiquetas del modelo.
 - trainning(): Realiza el entrenamiento del modelo con los datos preprocesados.
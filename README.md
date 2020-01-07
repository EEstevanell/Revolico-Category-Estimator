# Revolico-Category-Estimator

El objetivo principal de este proyecto es implementar estimadores que logren clasificar correctamente un bloque de texto en una de las categorías disponibles en el popular sitio de compra y venta cubana **Revolico**.

## Contenido

[TOC]

------



## Distribución del proyecto

El directorio del proyecto presenta la siguiente forma:

- docs
- src
  - crawler
    - websites
    - **spider.py**
    - **main_sp.py**
    
  - backend
  
  - data
  
  - engine
    - **base.py**
    - **LinearSVC.py**
    - **LogisticRegression.py**
    - **MultinomialNB.py**
    
  - frontend
  
  - **main.py**
  
  - **flask_init**
  
    ------
  
    
  
### Engine

El fichero **base.py** contiene la definición de un estimador **BaseAdvertiseClassifier** que hereda de **BaseEstimator** de **Scikit-Learn**. Dicha clase implementada permite una fácil integración de estimadores y vectorizadores para formar un *pipeline* de *machine learning* básico. Además, incluyen funciones para la evaluación de dichos modelos utilizando *k-fold cross validation*.

Los ficheros **LinearSVC.py**, **LogisticRegression.py** y **MultinomialNB.py** en *engine* constituyen estimadores customizados que implementan **BaseAdvertiseClassifier**.

------



### Main

El fichero **main.py** presenta una gama de funciones que permiten trabajar directamente con los clasificadores implementados en *engine*.

- ***plot_learning_curves*** presenta las curvas de aprendizaje de los estimadores implementados sobre un *corpus* determinado. El parámetro *k* representa la cantidad de *folds* que se utilizarán para la evaluación de los modelos (*k-fold cross validation*) con cada tamaño de *dataset*.
- ***get_strattified_data*** genera un arreglo de subconjuntos de un *dataset* con tamaño creciente. Cabe resaltar que la principal ventaja de este método es que en cada iteración del generador obtenemos un conjunto de datos balanceado con respecto a las clases (solo si es posible).
- ***fit*** permite entrenar los estimadores con un *dataset* determinado. Este proceso se ha optimizado para que haga cada ***fit*** independiente en paralelo.
- ***predict*** permite predecir para un conjunto de datos las clases correspondientes a cada uno. Se devuelve un arreglo con los resultados de cada estimador.

------



### flask_init

El fichero **flask_init.py** comprende el manejo del backend y el renderizado del frontend. Al ejecutar python flask_init.py, se crea un server que sirve por defecto en localhost:5000; al abrirlo en el navegador despliega la interfaz visual. En la primera vista podemos seleccionar entre entrenar el modelo o clasificar un texto. En caso de no seleccionar ningún entrenamiento, se realizará con el corpus que viene por defecto.
En la vista de entrenamiento hay 3 opciones posibles:

1. Default training: Entrenar con el corpus por defecto
2. Load training set from path: Introducir una dirección de un directorio donde se encuentre un corpus que desee usar
3. Craw Revolico from local server: Al dar click aquí aparecerá otra vista, en la cual hay que introducir el *seed* del *crawler* (donde va a comenzar a *crawlear*) y la cantidad de páginas a descargar.Es importante decir que el *crawler* está hecho para trabajar con el Revolico que viene en el paquete semanal, por lo que una sugerencia para poder utilizarlo es montar un http.server local en el directorio donde se encuentra el índice del sitio(Revolico)

En la vista de clasificar el texto, se brinda la opción de introducir un texto directamente(Classify text) o de introducir la dirección a un directorio con varios documentos de texto y clasificarlos todos.

------



### main_sp

El fichero **main_sp.py**  posee un único método que se encarga de correr el *crawler*.
- ***run_crawler*** Ejecuta el crawler. Recibe como parámetros la cantidad de sitios a *crawlear* y el *seed* desde el que comenzará el proceso. Al concluir quedarán guardados los datos como una tupla serializada (categoría, cuerpo del anuncio) dentro de la carpeta *websites*

## Requerimientos

- [x] Flask==1.0.2
- [x] matplotlib==3.0.3
- [x] Scrapy==1.7.3
- [x] nltk==3.4.5
- [x] numpy==1.16.3
- [x] scikit_learn==0.22.1

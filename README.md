# Revolico-Category-Estimator

El objetivo principal de este proyecto es implementar estimadores que logren clasificar correctamente un bloque de texto en una de las categorías disponibles en el popular sitio de compra y venta cubana **Revolico**.

## Distribución del proyecto

El directorio del proyecto presenta la siguiente forma:

- docs
- src
  - backend
  - data
  - engine
    * **base.py**
    * **LinearSVC.py**
    * **LogisticRegression.py**
    * **MultinomialNB.py**
  - frontend
  - **main.py**

### Engine

El fichero **base.py** contiene la definición de un estimador **BaseAdvertiseClassifier** que hereda de **BaseEstimator** de **Scikit-Learn**. Dicha clase implementada permite una fácil integración de estimadores y vectorizadores para formar un *pipeline* de *machine learning* básico. Además, incluyen funciones para la evaluación de dichos modelos utilizando *k-fold cross validation*.

Los ficheros **LinearSVC.py**, **LogisticRegression.py** y **MultinomialNB.py** en *engine* constituyen estimadores customizados que implementan **BaseAdvertiseClassifier**.

### Main

El fichero **main.py** presenta una gamas de funciones que permiten trabajar directamente con los clasificadores implementados en *engine*.

- ***plot_learning_curves*** presenta las curvas de aprendizaje de los estimadores implementados sobre un *corpus* determinado. El parámetro *k* representa la cantidad de *folds* que se utilizarán para la evaluación de los modelos (*k-fold cross validation*) con cada tamaño de *dataset*.
- ***get_strattified_data*** genera un arreglo de subconjuntos de un *dataset* con tamaño creciente. Cabe resaltar que la principal ventaja de este método es que en cada iteración del generador obtenemos un conjunto de datos balanceado con respecto a las clases (solo si es posible).
- ***fit*** permite entrenar los estimadores con un *dataset* determinado. Este proceso se ha optimizado para que haga cada ***fit*** independiente en paralelo.
- ***predict*** permite predecir para un conjunto de datos las clases correspondientes a cada uno. Se devuelve un arreglo con los resultados de cada estimador.

## Requerimientos

- [x] matplotlib == 3.0.3
- [x] nltk == 3.4.5
- [x] numpy == 1.16.3
- [x] scikit_learn == 0.22

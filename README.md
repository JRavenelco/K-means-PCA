# Análisis de Agrupamiento con K-means y Reducción de Dimensionalidad con PCA

Este proyecto implementa el algoritmo de agrupamiento K-means junto con Análisis de Componentes Principales (PCA) para el análisis exploratorio de datos. El objetivo es agrupar conjuntos de datos en grupos homogéneos y visualizar los resultados en un espacio reducido de dimensiones.

## Estructura del Proyecto

```
/
|-- data/                  # Datos de ejemplo
|-- notebooks/             # Jupyter notebooks con análisis
|-- src/                   # Código fuente
|   |-- kmeans.py          # Implementación del algoritmo K-means
|   |-- pca.py             # Implementación de PCA
|   |-- utils.py           # Funciones auxiliares
|   `-- visualization.py   # Visualización de resultados
|-- requirements.txt       # Dependencias de Python
|-- .gitignore
`-- README.md
```

## Tecnologías Utilizadas

*   **Lenguaje Principal**:
    *   Python 3.x

*   **Librerías Principales**:
    *   NumPy - Cálculos numéricos
    *   Pandas - Manipulación de datos
    *   Matplotlib - Visualización de datos
    *   Scikit-learn - Implementación de referencia para comparación

*   **Herramientas de Desarrollo**:
    *   Jupyter Notebook - Análisis interactivo
    *   Git - Control de versiones

## Instalación

### 1. Entorno Python

Se recomienda utilizar un entorno virtual para gestionar las dependencias de Python.

```bash
# Crear y activar un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# Instalar las dependencias
pip install -r requirements.txt
```

## Uso

### 1. Cargar y Preprocesar Datos

```python
from src.utils import load_data
from src.pca import PCA
from src.kmeans import KMeans

# Cargar datos
data = load_data('ruta/al/archivo.csv')

# Normalizar datos
data_normalized = (data - data.mean()) / data.std()
```

### 2. Aplicar PCA para Reducción de Dimensionalidad

```python
# Inicializar PCA
pca = PCA(n_components=2)


# Ajustar y transformar datos
data_pca = pca.fit_transform(data_normalized)
```

### 3. Aplicar K-means para Agrupamiento

```python
# Inicializar K-means
kmeans = KMeans(k=3, max_iters=100)

# Ajustar el modelo
clusters = kmeans.fit_predict(data_pca)
```

### 4. Visualizar Resultados

```python
from src.visualization import plot_clusters

# Visualizar clusters
plot_clusters(data_pca, clusters, kmeans.centroids)
```

## Algoritmos Implementados

### K-means

K-means es un algoritmo de agrupamiento que busca dividir un conjunto de datos en K grupos, donde cada observación pertenece al grupo cuya media es más cercana. El algoritmo itera entre dos pasos principales:

1. **Asignación**: Cada punto se asigna al centroide más cercano.
2. **Actualización**: Se actualiza la posición de los centroides como la media de los puntos asignados.

### Análisis de Componentes Principales (PCA)

PCA es una técnica de reducción de dimensionalidad que transforma un conjunto de variables correlacionadas en un conjunto más pequeño de variables no correlacionadas llamadas componentes principales. Los componentes principales son combinaciones lineales de las variables originales que capturan la mayor varianza posible en los datos.

## Contribuciones

¡Las contribuciones son bienvenidas! Siéntete libre de enviar un pull request o abrir un issue.

1.  Haz un Fork del proyecto
2.  Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3.  Haz commit de tus cambios (`git commit -m 'Añadir nueva característica'`)
4.  Sube los cambios a tu rama (`git push origin feature/nueva-caracteristica`)
5.  Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.

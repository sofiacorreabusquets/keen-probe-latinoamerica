# keen-probe-latinoamerica

Proyecto de entrenamiento de una sonda (probe) MLP para predecir scores basándose en representaciones vectoriales normalizadas, basado en el código original de [keen_estimating_knowledge_in_llms](https://github.com/dhgottesman/keen_estimating_knowledge_in_llms).

## Estructura del Repositorio

```
keen-probe-latinoamerica/
├── data/                   # Directorio de datos
│   ├── filter_vectors.ipynb       # Notebook para filtrar vectores según entidades
│   ├── filtered_vectors.npz       # Vectores normalizados filtrados (X)
│   ├── score_by_entity.json       # Scores por entidad (y)
│   └── state_b8d56cb4aa70_hs.npz  # Vectores originales (gitignored por tamaño)
├── hyperparams.py           # Hiperparámetros del modelo (batch size, optimizer, learning rate, max iterations)
├── mlp_regressor.py        # MLPRegressor de código fuente KEEN debuggeado
├── probe.ipynb             # Notebook principal para entrenar la sonda
└── models/                 # Directorio de modelos entrenados
    └── mlp_regressor_state_dict.pth  # Pesos del modelo entrenado
```

## Archivos Principales

### `hyperparams.py`
Contiene los hiperparámetros de entrenamiento:
- `BATCH_SIZE = 32`
- `OPTIM = "adam"`
- `LR = 0.01`
- `MAX_ITER = 100`

### `mlp_regressor.py`
MLPRegressor de código fuente KEEN debuggeado, que incluye:
- Arquitectura de red neuronal simple (una capa lineal + sigmoid)
- Función de pérdida MSE
- Entrenamiento con validación
- Métricas de evaluación (Pearson correlation)
- Guardado de mejores pesos durante el entrenamiento

### `probe.ipynb`
Notebook principal con el flujo completo:
1. **Carga de datos**: Lee vectores normalizados desde `data/filtered_vectors.npz`
2. **Preparación de labels**: Mapea scores desde `data/score_by_entity.json` usando los índices
3. **Split de datos**: Divide en train/test (80/20)
4. **Entrenamiento**: Entrena el MLPRegressor con los hiperparámetros configurados
5. **Guardado**: Exporta el modelo entrenado a `models/mlp_regressor_state_dict.pth`

### `data/filter_vectors.ipynb`
Notebook de preprocesamiento que:
- Carga el archivo original `state_b8d56cb4aa70_hs.npz` 
- Filtra vectores según las entidades presentes en `score_by_entity.json`
- Genera `filtered_vectors.npz` con los vectores filtrados

## Datos

### Formato de Datos
- **`filtered_vectors.npz`**: Contiene:
  - `representations_normalized`: Array NumPy con vectores de representación normalizados
  - `row_indices`: Índices que mapean a las entidades en `score_by_entity.json`

- **`score_by_entity.json`**: Lista de objetos con estructura:
  ```json
  {
    "entidad": "nombre_entidad",
    "vector_index": 123,
    "score": 0.75
  }
  ```


## Características Técnicas

- Entrenamiento determinístico (seed fijo en 42)
- Uso de GPU mediante CUDA
- Validación con Pearson correlation
- Guardado automático de mejores pesos durante entrenamiento
- Sin logging de WandB (removido del código original)

## Requisitos

- PyTorch
- NumPy
- pandas
- scikit-learn
- scipy

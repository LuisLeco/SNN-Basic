# SNN-Basic

Proyecto para entrenar y evaluar diferentes codificadores y decodificadores para las Spiking Neural Networks (SNN) .

**Resumen:**
- Código en Python que implementa encoders, decoders, una arquitectura de dos capas LIF, y un flujo de experimentos paralelos.
- Genera un CSV con los resultados de las configuraciones probadas.

**Requisitos**
- Python 3.8+
- torch
- torchvision
- snntorch

Instalación rápida:

```bash
python -m pip install torch torchvision snntorch
```

**Estructura de archivos**
- [Main.py](Main.py) — Script principal que lanza experimentos en paralelo y guarda resultados.
- [Experiment.py](Experiment.py) — Clase `SNNExperiment` que orquesta dataset, encoder, decoder, arquitectura y trainer.
- [Architecture.py](Architecture.py) — Definición de la arquitectura `TwoLayerSNN` (capas totalmente conectadas con neuronas LIF).
- [Encoding.py](Encoding.py) — Encoders disponibles (rate, ttfs, direct, delta, MW, SF, Deterministic, ...).
- [Decoding.py](Decoding.py) — Decoders disponibles (rate, latency, first_spike, population_rate, rank_order, AllDecoders).
- [Datasets.py](Datasets.py) — Carga MNIST con `DataLoader`.
- [Trainer.py](Trainer.py) — Ciclo de entrenamiento y evaluación.

**Uso**

Por defecto `Main.py` lanza experimentos en paralelo según la lista `num_steps`, 'encoder' y `decoders` definida en su cabecera. Ejemplo:

```bash
python Main.py
```

El script guardará un fichero CSV llamado `results <encoder>.csv` (por ejemplo `results ttfs.csv`) con las columnas:
- `encoder`, `decoder`, `num_steps`, `accuracy`, `numero de prueba`.

Para modificar configuraciones (encoder, decoders, num_steps, parámetros de entrenamiento), edita la variable `config_base` y las listas `num_steps`, `decoders` en [Main.py](Main.py).

**Configuración importante en `config_base`**
- `dataset`: 'MNIST' (actualmente soportado).
- `encoder`: Opciones: `poisson`, `rate`, `ttfs`, `direct`, `delta`, `MW`, `SF`, `Deterministic`.
- `decoder`: `rate`, `latency`, `first_spike`, `population_rate`, `rank_order`, `all`.
- `architecture`: actualmente `TwoLayerSNN`.
- `num_steps`, `batch_size`, `num_hidden`, `lr`, `num_epochs`, `eval_freq`, etc.

**Notas y recomendaciones**
- El proyecto usa GPU si está disponible; `Experiment` selecciona automáticamente `cuda` o `cpu`.
- `AllDecoders` devuelve una lista de salidas; el `Trainer` y `Main` ya manejan ambos casos (decoder único o múltiples).
- Ajusta `max_workers` en [Main.py](Main.py) si tu máquina tiene menos núcleos.
- Los datasets se descargan automáticamente en la ruta `data/mnist` (configurable via `config_base['data_path']`).

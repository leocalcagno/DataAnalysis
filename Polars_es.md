# Descubriendo Polars: La Alternativa Rápida a Pandas para Data Science (Parte 1/3)

**Por Leonardo Calcagno**  
*Fecha: Diciembre 2025*  

En un mundo donde los datos crecen a ritmos exponenciales –pensemos en petabytes generados diariamente por IoT, redes sociales y transacciones financieras–, las herramientas tradicionales como Pandas comienzan a mostrar sus grietas. Como Data Analyst, he visto cómo Pandas, aunque intuitivo, se queda corto en rendimiento y eficiencia de memoria cuando los datasets superan los gigabytes. Aquí es donde entra Polars: una biblioteca revolucionaria que no solo acelera tus workflows, sino que redefine cómo procesamos datos en Python.

En esta serie de tres artículos, exploraremos Polars desde cero hasta sus capacidades avanzadas. Basado en guías  y mi propia práctica en proyectos reales, te guiaré para que migres y multipliques tu productividad. Si eres Data Engineer, Scientist o simplemente un entusiasta de Python, esta serie te equipará para manejar datos a escala. ¡Empecemos con los fundamentos!

### ¿Por Qué Polars? Las Limitaciones de Pandas y la Revolución en Rust

Pandas ha sido el rey indiscutible del análisis de datos en Python desde 2008. Su API amigable, basada en DataFrames, permite manipular datos con facilidad. Sin embargo, en 2025, con datasets que fácilmente superan los 100 millones de filas, sus debilidades son evidentes:

- **Rendimiento single-threaded**: La mayoría de operaciones en Pandas usan un solo núcleo de CPU, lo que ralentiza procesos en máquinas modernas con múltiples núcleos.
- **Consumo de memoria alto**: Almacena datos como objetos Python, lo que infla el uso de RAM (hasta 4x más que formatos optimizados).
- **Ejecución eager**: Cada operación se ejecuta inmediatamente, sin optimizaciones globales, lo que desperdicia recursos en pipelines complejos.

Polars, desarrollado en Rust y lanzado en 2021, aborda estos problemas de raíz. Construido sobre Apache Arrow –un formato columnar en memoria que minimiza copias innecesarias–, Polars ofrece paralelización automática, lazy evaluation (evaluación perezosa) y un optimizador de consultas inspirado en bases de datos como SQL. El resultado: velocidades 5-50x superiores a Pandas en operaciones comunes, con un uso de memoria 2-4x menor.

Aquí una comparación rápida basada en benchmarks estándar (datasets como Titanic y Diamonds):

| Característica       | Polars                          | Pandas                     |
|----------------------|---------------------------------|----------------------------|
| Motor de ejecución  | Rust + Apache Arrow + Optimizador | Python + NumPy            |
| Paralelización      | Automática (multi-core)        | Mayormente single-thread  |
| Uso de memoria      | Bajo (formato columnar)        | Alto (objetos Python)     |
| Lazy Evaluation     | Nativa y optimizada            | Experimental (df.delayed) |
| Velocidad típica    | 5-20x más rápido               | Referencia                |

Casos de uso ideales para Polars incluyen pipelines ETL con datos >1GB, análisis exploratorio {EDA} en datasets grandes, feature engineering a escala y procesamiento en producción donde la eficiencia es clave. En mi experiencia, migrar a Polars ha reducido tiempos de ejecución significativamente.

### Instalación y Setup: Listo en Minutos

Comenzar con Polars es sencillo. Recomiendo usar conda para binarios optimizados, pero pip funciona igual de bien en la mayoría de plataformas.

```bash
# Vía conda (recomendado para producción)
conda install -c conda-forge polars

# O vía pip
pip install polars
```

Una vez instalado, importa la biblioteca y configura el entorno para una mejor visualización:

```python
import polars as pl
import time  # Para benchmarks
import numpy as np
import pandas as pd  # Para comparaciones

# Configuración recomendada
pl.Config.set_tbl_rows(20)  # Mostrar hasta 20 filas
pl.Config.set_tbl_cols(12)  # Mostrar hasta 12 columnas
pl.Config.set_fmt_str_lengths(50)  # Longitud máxima de strings
pl.Config.set_tbl_width_chars(120)  # Ancho de tabla

print(f"Polars versión: {pl.__version__}")
```

Esta configuración asegura que tus outputs sean legibles, especialmente en notebooks Jupyter o VS Code.

### Carga y Exploración Básica: Más Rápido que Nunca

Polars soporta múltiples fuentes: diccionarios, CSV, Parquet, incluso desde Pandas. Usemos datasets públicos como Titanic (891 filas) para ilustrar.

```python
# Carga desde URL (eager mode)
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
start = time.time()
df_titanic = pl.read_csv(TITANIC_URL)
print(f"Tiempo de carga: {time.time() - start:.4f}s")
print(df_titanic.head(3))
```

Salida típica:
```
Load Time: 0.3230s
shape: (3, 12)
┌────────────┬──────────┬────────┬────────────┬────────┬──────┬───────┬───────┬───────────┬─────────┬───────┬──────────┐
│ PassengerI ┆ Survived ┆ Pclass ┆ Name       ┆ Sex    ┆ Age  ┆ SibSp ┆ Parch ┆ Ticket    ┆ Fare    ┆ Cabin ┆ Embarked │
│ d          ┆ ---      ┆ ---    ┆ ---        ┆ ---    ┆ ---  ┆ ---   ┆ ---   ┆ ---       ┆ ---     ┆ ---   ┆ ---      │
│ ---        ┆ i64      ┆ i64    ┆ str        ┆ str    ┆ f64  ┆ i64   ┆ i64   ┆ str       ┆ f64     ┆ str   ┆ str      │
│ i64        ┆          ┆        ┆            ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
╞════════════╪══════════╪════════╪════════════╪════════╪══════╪═══════╪═══════╪═══════════╪═════════╪═══════╪══════════╡
│ 1          ┆ 0        ┆ 3      ┆ Braund,    ┆ male   ┆ 22.0 ┆ 1     ┆ 0     ┆ A/5 21171 ┆ 7.25    ┆ null  ┆ S        │
│            ┆          ┆        ┆ Mr. Owen   ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│            ┆          ┆        ┆ Harris     ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│ 2          ┆ 1        ┆ 1      ┆ Cumings,   ┆ female ┆ 38.0 ┆ 1     ┆ 0     ┆ PC 17599  ┆ 71.2833 ┆ C85   ┆ C        │
│            ┆          ┆        ┆ Mrs. John  ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│            ┆          ┆        ┆ Bradley    ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│            ┆          ┆        ┆ (Florence  ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│            ┆          ┆        ┆ Briggs     ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│            ┆          ┆        ┆ Thayer…    ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
│ 3          ┆ 1        ┆ 3      ┆ Heikkinen, ┆ female ┆ 26.0 ┆ 0     ┆ 0     ┆ STON/O2.  ┆ 7.925   ┆ null  ┆ S        │
│            ┆          ┆        ┆ Miss.      ┆        ┆      ┆       ┆       ┆ 3101282   ┆         ┆       ┆          │
│            ┆          ┆        ┆ Laina      ┆        ┆      ┆       ┆       ┆           ┆         ┆       ┆          │
└────────────┴──────────┴────────┴────────────┴────────┴──────┴───────┴───────┴───────────┴─────────┴───────┴──────────┘

```

Exploración es intuitiva:
```python
print(f"Dimensiones: {df_titanic.shape}")
print(f"Columnas: {df_titanic.columns}")
print(df_titanic.dtypes)
print(df_titanic.describe())
print(df_titanic.null_count())
print(df_titanic.tail(3))
```

```python
Dimension: (891, 12)
Columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
[Int64, Int64, Int64, String, String, Float64, Int64, Int64, String, Float64, String, String]
shape: (9, 13)
┌─────────┬─────────┬─────────┬─────────┬─────────┬────────┬───┬─────────┬─────────┬─────────┬────────┬───────┬────────┐
│ statist ┆ Passeng ┆ Survive ┆ Pclass  ┆ Name    ┆ Sex    ┆ … ┆ SibSp   ┆ Parch   ┆ Ticket  ┆ Fare   ┆ Cabin ┆ Embark │
│ ic      ┆ erId    ┆ d       ┆ ---     ┆ ---     ┆ ---    ┆   ┆ ---     ┆ ---     ┆ ---     ┆ ---    ┆ ---   ┆ ed     │
│ ---     ┆ ---     ┆ ---     ┆ f64     ┆ str     ┆ str    ┆   ┆ f64     ┆ f64     ┆ str     ┆ f64    ┆ str   ┆ ---    │
│ str     ┆ f64     ┆ f64     ┆         ┆         ┆        ┆   ┆         ┆         ┆         ┆        ┆       ┆ str    │
╞═════════╪═════════╪═════════╪═════════╪═════════╪════════╪═══╪═════════╪═════════╪═════════╪════════╪═══════╪════════╡
│ count   ┆ 891.0   ┆ 891.0   ┆ 891.0   ┆ 891     ┆ 891    ┆ … ┆ 891.0   ┆ 891.0   ┆ 891     ┆ 891.0  ┆ 204   ┆ 889    │
│ null_co ┆ 0.0     ┆ 0.0     ┆ 0.0     ┆ 0       ┆ 0      ┆ … ┆ 0.0     ┆ 0.0     ┆ 0       ┆ 0.0    ┆ 687   ┆ 2      │
│ unt     ┆         ┆         ┆         ┆         ┆        ┆   ┆         ┆         ┆         ┆        ┆       ┆        │
│ mean    ┆ 446.0   ┆ 0.38383 ┆ 2.30864 ┆ null    ┆ null   ┆ … ┆ 0.52300 ┆ 0.38159 ┆ null    ┆ 32.204 ┆ null  ┆ null   │
│         ┆         ┆ 8       ┆ 2       ┆         ┆        ┆   ┆ 8       ┆ 4       ┆         ┆ 208    ┆       ┆        │
│ std     ┆ 257.353 ┆ 0.48659 ┆ 0.83607 ┆ null    ┆ null   ┆ … ┆ 1.10274 ┆ 0.80605 ┆ null    ┆ 49.693 ┆ null  ┆ null   │
│         ┆ 842     ┆ 2       ┆ 1       ┆         ┆        ┆   ┆ 3       ┆ 7       ┆         ┆ 429    ┆       ┆        │
│ min     ┆ 1.0     ┆ 0.0     ┆ 1.0     ┆ Abbing, ┆ female ┆ … ┆ 0.0     ┆ 0.0     ┆ 110152  ┆ 0.0    ┆ A10   ┆ C      │
│         ┆         ┆         ┆         ┆ Mr.     ┆        ┆   ┆         ┆         ┆         ┆        ┆       ┆        │
│         ┆         ┆         ┆         ┆ Anthony ┆        ┆   ┆         ┆         ┆         ┆        ┆       ┆        │
│ 25%     ┆ 224.0   ┆ 0.0     ┆ 2.0     ┆ null    ┆ null   ┆ … ┆ 0.0     ┆ 0.0     ┆ null    ┆ 7.925  ┆ null  ┆ null   │
│ 50%     ┆ 446.0   ┆ 0.0     ┆ 3.0     ┆ null    ┆ null   ┆ … ┆ 0.0     ┆ 0.0     ┆ null    ┆ 14.454 ┆ null  ┆ null   │
│         ┆         ┆         ┆         ┆         ┆        ┆   ┆         ┆         ┆         ┆ 2      ┆       ┆        │
│ 75%     ┆ 669.0   ┆ 1.0     ┆ 3.0     ┆ null    ┆ null   ┆ … ┆ 1.0     ┆ 0.0     ┆ null    ┆ 31.0   ┆ null  ┆ null   │
│ max     ┆ 891.0   ┆ 1.0     ┆ 3.0     ┆ van Mel ┆ male   ┆ … ┆ 8.0     ┆ 6.0     ┆ WE/P    ┆ 512.32 ┆ T     ┆ S      │
...
│ 891         ┆ 0        ┆ 3      ┆ Dooley,    ┆ male   ┆ 32.0 ┆ 0     ┆ 0     ┆ 370376     ┆ 7.75  ┆ null  ┆ Q        │
│             ┆          ┆        ┆ Mr.        ┆        ┆      ┆       ┆       ┆            ┆       ┆       ┆          │
│             ┆          ┆        ┆ Patrick    ┆        ┆      ┆       ┆       ┆            ┆       ┆       ┆          │
└─────────────┴──────────┴────────┴────────────┴────────┴──────┴───────┴───────┴────────────┴───────┴───────┴──────────┘
```

Esto revela estadísticas descriptivas, tipos de datos y valores nulos en segundos. Comparado con Pandas, Polars es notablemente más eficiente incluso en cargas pequeñas.




### Conclusión y Teaser para la Parte 2

Polars no es solo una herramienta; es un cambio de paradigma que te permite enfocarte en insights en lugar de esperar que tu código termine. En mi carrera, ha transformado workflows lentos en procesos ágiles, ahorrando horas valiosas.

En la **Parte 2**, profundizaremos en transformaciones eficientes: selección, filtrado, group_by y creación de columnas. 

*Enlaces útiles: [Documentación oficial de Polars](https://docs.pola.rs), [GitHub de Polars](https://github.com/pola-rs/polars).*

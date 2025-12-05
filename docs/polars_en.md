# Discovering Polars: The Fast Alternative to Pandas for Data Science (Part 1/3)

**By Leonardo Calcagno**  
*Date: December 2025*  

In a world where data grows at exponential rates—think of petabytes generated daily by IoT, social media, and financial transactions—traditional tools like Pandas begin to show their cracks. As a Data Analyst, I've seen how Pandas, while intuitive, falls short in performance and memory efficiency when datasets exceed gigabytes. This is where Polars comes in: a revolutionary library that not only accelerates your workflows but redefines how we process data in Python.

In this three-part series, we'll explore Polars from scratch to its advanced capabilities. Based on guides and my own practice in real projects, I'll guide you to migrate and multiply your productivity. Whether you're a Data Engineer, Scientist, or simply a Python enthusiast, this series will equip you to handle data at scale. Let's start with the fundamentals!

### Why Polars? Pandas Limitations and the Rust Revolution

Pandas has been the undisputed king of data analysis in Python since 2008. Its friendly API, based on DataFrames, allows easy data manipulation. However, in 2025, with datasets that easily exceed 100 million rows, its weaknesses are evident:

- **Single-threaded performance**: Most operations in Pandas use a single CPU core, which slows down processes on modern machines with multiple cores.
- **High memory consumption**: Stores data as Python objects, which inflates RAM usage (up to 4x more than optimized formats).
- **Eager execution**: Each operation executes immediately, without global optimizations, wasting resources in complex pipelines.

Polars, developed in Rust and launched in 2021, addresses these problems at their root. Built on Apache Arrow—a columnar in-memory format that minimizes unnecessary copies—Polars offers automatic parallelization, lazy evaluation, and a query optimizer inspired by databases like SQL. The result: speeds 5-50x higher than Pandas in common operations, with memory usage 2-4x lower.

Here's a quick comparison based on standard benchmarks (datasets like Titanic and Diamonds):

| Feature             | Polars                            | Pandas                     |
|---------------------|-----------------------------------|----------------------------|
| Execution engine    | Rust + Apache Arrow + Optimizer   | Python + NumPy             |
| Parallelization     | Automatic (multi-core)            | Mostly single-thread       |
| Memory usage        | Low (columnar format)             | High (Python objects)      |
| Lazy Evaluation     | Native and optimized              | Experimental (df.delayed)  |
| Typical speed       | 5-20x faster                      | Reference                  |

Ideal use cases for Polars include ETL pipelines with data >1GB, exploratory data analysis (EDA) on large datasets, feature engineering at scale, and production processing where efficiency is key. In my experience, migrating to Polars has significantly reduced execution times.

### Installation and Setup: Ready in Minutes

Getting started with Polars is simple. I recommend using conda for optimized binaries, but pip works just as well on most platforms.

```bash
# Via conda (recommended for production)
conda install -c conda-forge polars

# Or via pip
pip install polars
```

Once installed, import the library and configure the environment for better visualization:

```python
import polars as pl
import time  # For benchmarks
import numpy as np
import pandas as pd  # For comparisons

# Recommended configuration
pl.Config.set_tbl_rows(20)  # Show up to 20 rows
pl.Config.set_tbl_cols(12)  # Show up to 12 columns
pl.Config.set_fmt_str_lengths(50)  # Maximum string length
pl.Config.set_tbl_width_chars(120)  # Table width

print(f"Polars version: {pl.__version__}")
```

This configuration ensures your outputs are readable, especially in Jupyter notebooks or VS Code.

### Loading and Basic Exploration: Faster Than Ever

Polars supports multiple sources: dictionaries, CSV, Parquet, even from Pandas. Let's use public datasets like Titanic (891 rows) to illustrate.

```python
# Load from URL (eager mode)
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
start = time.time()
df_titanic = pl.read_csv(TITANIC_URL)
print(f"Load time: {time.time() - start:.4f}s")
print(df_titanic.head(3))
```

Typical output:
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

Exploration is intuitive:
```python
print(f"Dimensions: {df_titanic.shape}")
print(f"Columns: {df_titanic.columns}")
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

This reveals descriptive statistics, data types, and null values in seconds. Compared to Pandas, Polars is noticeably more efficient even on small loads.

### Conclusion 

Polars is not just a tool; it's a paradigm shift that allows you to focus on insights instead of waiting for your code to finish. In my career, it has transformed slow workflows into agile processes, saving valuable hours.

In **Part 2**, we'll dive deeper into efficient transformations: selection, filtering, group_by, and column creation.

*Useful links: [Official Polars Documentation](https://docs.pola.rs), [Polars GitHub](https://github.com/pola-rs/polars).*
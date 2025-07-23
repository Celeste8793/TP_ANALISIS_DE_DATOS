import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CARGA Y LIMPIEZA DE DATOS
df = pd.read_csv("encuesta_bienestar.csv")

# Eliminar valores negativos o nulos en variables clave
df = df[df["screen_time"] >= 0]
df = df.dropna(subset=["age", "hours_sleep", "exercise_days", "happiness_level"])

# 2. CREACIÓN DE VARIABLES DERIVADAS

# Ratio de pantalla/sueño
df["screen_sleep_ratio"] = df["screen_time"] / df["hours_sleep"]

# Índice de hábitos saludables (0 a 3)
df["habito_saludable"] = (
    (df["hours_sleep"] >= 7).astype(int) +
    (df["exercise_days"] >= 3).astype(int) +
    (df["screen_time"] <= 6).astype(int)
)

# Grupo de riesgo: pantalla alta + poco sueño + estrés alto
df["grupo_riesgo"] = (
    (df["screen_time"] > 8) &
    (df["hours_sleep"] < 6) &
    (df["stress_level"] >= 8)
).astype(int)

# 3. ANÁLISIS UNIVARIADO

variables_numericas = ["age", "hours_sleep", "exercise_days", "screen_time", "happiness_level"]

for var in variables_numericas:
    sns.histplot(df[var], kde=True, color="skyblue")
    plt.title(f"Distribución de {var}")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")
    plt.show()

# 4. ANÁLISIS BIVARIADO

# Correlación
sns.heatmap(df[variables_numericas + ["stress_level"]].corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

# Dispersión: sueño vs felicidad
sns.scatterplot(data=df, x="hours_sleep", y="happiness_level", hue="gender")
plt.title("Relación entre sueño y felicidad")
plt.show()

# 5. COMPARACIONES POR GÉNERO Y PAÍS

# Género
sns.boxplot(data=df, x="gender", y="happiness_level")
plt.title("Felicidad por género")
plt.show()

# País
sns.barplot(data=df, x="country", y="happiness_level", errorbar="sd")
plt.title("Felicidad promedio por país")
plt.xticks(rotation=45)
plt.show()

# 6. EXPORTACIÓN DE DATOS LIMPIOS
df.to_csv("encuesta_bienestar_limpio.csv", index=False)

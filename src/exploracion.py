import pandas as pd

ENFERMEDADES = ["DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
                "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO"]
VALIDOS = {"USMER": {1, 2}, "MEDICAL_UNIT": set(range(1, 14)), "SEX": {1, 2},
           "PATIENT_TYPE": {1, 2}, "PNEUMONIA": {1, 2}, "ICU": {0, 1},
           **{c: {0, 1} for c in ENFERMEDADES}}


def titulo(t):
    print(f"\n=== {t} ===")


df = pd.read_csv("dataset/Covid_Dataset.csv", dtype=str, keep_default_na=False)
df.columns = df.columns.str.strip().str.upper()
df = df.apply(lambda c: c.str.strip())
num = [c for c in df.columns if c != "DATE_DIED"]
df[num] = df[num].apply(pd.to_numeric, errors="coerce")

titulo("1. Estructura (por qué leer como texto y convertir)")
print("Filas y columnas:", df.shape)
print("Valores no numéricos:\n", df[num].isna().sum()[lambda s: s > 0])

titulo("2. CLASIFFICATION_FINAL (por qué filtrar 1, 2 y 3)")
print(df["CLASIFFICATION_FINAL"].value_counts().sort_index())
print("Se conservan:", df["CLASIFFICATION_FINAL"].isin([1, 2, 3]).sum())
df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].reset_index(drop=True)

titulo("3. DATE_DIED (por qué crear MUERTO)")
sin_fecha = df["DATE_DIED"] == "9999-99-99"
print("Sin fecha (vivos):", sin_fecha.sum(), "| Con fecha:", (~sin_fecha).sum())
fechas = pd.to_datetime(df.loc[~sin_fecha, "DATE_DIED"], dayfirst=True, errors="coerce")
print("Fechas ilegibles:", fechas.isna().sum())

titulo("4. AGE (por qué solo corregir fuera de 0-120)")
print(df["AGE"].describe())
print("Fuera de 0-120:", (~df["AGE"].between(0, 120)).sum())

titulo("5. Códigos inválidos (por qué imputar por moda)")
for col, ok in VALIDOS.items():
    print(f"{col}: {df[col].value_counts().sort_index().to_dict()} "
          f"| inválidos: {(~df[col].isin(ok)).sum()}")

titulo("6. INTUBED vs PATIENT_TYPE (por qué 3 = no aplica en ambulatorios)")
print(pd.crosstab(df["PATIENT_TYPE"], df["INTUBED"]))

titulo("7. PREGNANT vs SEX (por qué 3 en hombres y moda en mujeres)")
print(pd.crosstab(df["SEX"], df["PREGNANT"]))
nina = (df["SEX"] == 1) & (df["AGE"] < 10)
print("Niñas menores de 10:", nina.sum(), "| con PREGNANT=1:", (df.loc[nina, "PREGNANT"] == 1).sum())

titulo("8. ICU vs PATIENT_TYPE (advertencia de redundancia)")
print(pd.crosstab(df["PATIENT_TYPE"], df["ICU"]))

titulo("9. Duplicados (por qué no se eliminaron / riesgo de fuga)")
cols = [c for c in df.columns if c != "DATE_DIED"]
dup = df.duplicated(subset=cols).sum()
print(f"Filas repetidas: {dup} ({dup / len(df) * 100:.1f}%)")

titulo("10. Balance de MUERTO (por qué se creó el CSV balanceado)")
df["MUERTO"] = (~sin_fecha).astype(int)
print(df["MUERTO"].value_counts())
print(f"% fallecidos: {df['MUERTO'].mean() * 100:.2f}%")
print("\nMortalidad por tipo de paciente:")
print(df.groupby("PATIENT_TYPE")["MUERTO"].mean().round(3))
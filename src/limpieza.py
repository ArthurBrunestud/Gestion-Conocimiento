from pathlib import Path
import pandas as pd

ENTRADA = Path("dataset/Covid_Dataset.csv")
SEMILLA = 42
SIN_FECHA = "9999-99-99"

ENFERMEDADES = ["DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
                "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO"]
VALIDOS = {"USMER": {1, 2}, "MEDICAL_UNIT": set(range(1, 14)), "SEX": {1, 2},
           "PATIENT_TYPE": {1, 2}, "PNEUMONIA": {1, 2}, "ICU": {0, 1},
           **{c: {0, 1} for c in ENFERMEDADES}}


def moda(serie, validos):
    """Valor más frecuente entre los válidos (en empate, el menor)."""
    return int(serie[serie.isin(validos)].mode().min())


df = pd.read_csv(ENTRADA, dtype=str, encoding="utf-8-sig", keep_default_na=False)
df.columns = df.columns.str.strip().str.upper()
df = df.apply(lambda c: c.str.strip())
numericas = [c for c in df.columns if c != "DATE_DIED"]
df[numericas] = df[numericas].apply(pd.to_numeric, errors="coerce")

# 2) Solo COVID confirmado
df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].reset_index(drop=True)

# 3) Limpieza por moda
df.loc[~df["AGE"].between(0, 120), "AGE"] = moda(df["AGE"], range(121))
for col, ok in VALIDOS.items():
    df.loc[~df[col].isin(ok), col] = moda(df[col], ok)

hosp = df["PATIENT_TYPE"] == 2
df.loc[~hosp, "INTUBED"] = 3                      # ambulatorio: no aplica
df.loc[hosp & ~df["INTUBED"].isin([1, 2]), "INTUBED"] = moda(df.loc[hosp, "INTUBED"], {1, 2})

elegible = (df["SEX"] == 1) & (df["AGE"] >= 10)
moda_emb = moda(df.loc[elegible, "PREGNANT"], {1, 2})
df.loc[df["SEX"] == 2, "PREGNANT"] = 3            # hombre: no aplica
df.loc[(df["SEX"] == 1) & (df["AGE"] < 10), "PREGNANT"] = 2
df.loc[elegible & ~df["PREGNANT"].isin([1, 2]), "PREGNANT"] = moda_emb

# 4) Variable objetivo y tipos
df["MUERTO"] = (df["DATE_DIED"] != SIN_FECHA).astype(int)
df[numericas] = df[numericas].astype(int)

# 5) Guardar natural y balanceado (50/50)
df.to_csv(ENTRADA.with_name("Covid_Dataset_Limpio_V4_natural.csv"), index=False, encoding="utf-8-sig")
muertos = df[df["MUERTO"] == 1]
vivos = df[df["MUERTO"] == 0].sample(n=len(muertos), random_state=SEMILLA)
bal = pd.concat([muertos, vivos]).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
bal.to_csv(ENTRADA.with_name("Covid_Dataset_Limpio_V4_balanceado.csv"), index=False, encoding="utf-8-sig")

print(f"Natural: {len(df):,} filas, {df['MUERTO'].mean()*100:.2f}% fallecidos")
print(f"Balanceado: {len(bal):,} filas")
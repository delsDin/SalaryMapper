# SalaryMapper 🎯

> Modèle de régression linéaire **from scratch** pour la prédiction salariale en fonction des années d'expérience — implémenté par descente de gradient batch avec normalisation z-score, validation croisée k-fold et analyse statistique complète des résidus.

---

## Aperçu

SalaryMapper est un projet d'apprentissage automatique pédagogique qui construit une régression linéaire simple **sans bibliothèque de machine learning**, à partir des équations mathématiques fondamentales. L'objectif est double : produire un modèle performant et valider chaque étape de l'implémentation par comparaison avec la référence `scikit-learn`.

**Équation finale :**

```
ŷ = 9 420.81 × x_exp + 24 396.10
```

Chaque année d'expérience supplémentaire est associée en moyenne à une hausse salariale de **~9 421 $**.

---

## Résultats

| Métrique | Train | Test (hold-out) | CV 5-fold (moy ± σ) |
|----------|-------|-----------------|----------------------|
| R²       | 0.9645 | 0.9024         | 0.9491 ± 0.0122      |
| RMSE     | 5 206 $ | 7 060 $       | 6 142 ± 620 $        |
| MAE      | 4 223 $ | 6 286 $       | 4 960 ± 500 $        |

> **Delta R² (train − test) = 0.062** → pas d'overfitting significatif.

---

## Structure du projet

```
SalaryMapper/
├── main.ipynb            # Notebook complet — 15 sections, 33 cellules
├── Salary_dataset.csv    # Dataset (30 observations)
└── README.md             # Ce fichier
```

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | Salary Dataset (domaine public) |
| Observations | 30 |
| Variable prédictive X | `YearsExperience` (1.2 – 10.6 ans) |
| Variable cible Y | `Salary` (37 732 $ – 122 392 $) |
| Valeurs manquantes | 0 |
| Corrélation de Pearson | **r = 0.9782** |

---

## Méthodologie

### 1. Préparation des données

- Split train/test **80/20** (seed = 42) → 24 observations en train, 6 en test
- **Normalisation z-score** calculée exclusivement sur le train (pas de data leakage)

```python
x_n = (x - μ_x) / σ_x
y_n = (y - μ_y) / σ_y
```

### 2. Implémentation from scratch

**Fonction de coût** (MSE/2 — facilite le calcul des gradients) :

```python
J(w, b) = (1 / 2m) × Σ (w·x_i + b − y_i)²
```

**Descente de gradient batch** :

```python
dw = (1/m) × Σ (ŷ_i − y_i) · x_i
db = (1/m) × Σ (ŷ_i − y_i)

w ← w − lr · dw
b ← b − lr · db
```

**Hyperparamètres :**

| Paramètre | Valeur |
|-----------|--------|
| Taux d'apprentissage `lr` | 0.01 |
| Itérations max | 10 000 |
| Tolérance (critère d'arrêt) | 1e-9 |

### 3. Dénormalisation

```python
w_orig = w_n × (σ_y / σ_x)
b_orig = μ_y + σ_y × b_n − w_orig × μ_x
```

### 4. Validation croisée k-fold

K-fold à 5 plis sur l'ensemble complet (n=30). Pour chaque pli, le modèle est ré-entraîné from scratch avec normalisation indépendante, afin d'obtenir une estimation robuste sans biais d'échantillonnage.

### 5. Validation vs scikit-learn

| Paramètre | From Scratch | Sklearn | Δ |
|-----------|-------------|---------|---|
| w (pente) | 9 420.81 | 9 421.04 | < 0.001 ✓ |
| b (biais)  | 24 396.10 | 24 396.27 | < 0.001 ✓ |

### 6. Analyse des résidus

Vérification des 3 hypothèses OLS :

| Hypothèse | Test | Résultat |
|-----------|------|----------|
| Normalité | Shapiro-Wilk | p > 0.05 ✓ |
| Autocorrélation | Durbin-Watson | ~2.05 ✓ (dans [1.5, 2.5]) |
| Homoscédasticité | Scale-Location plot | Variance stable ✓ |

### 7. Intervalles de confiance (IC 95%)

Via la distribution de Student (df = n − 2 = 22) :

```
w : 9 420.81  IC95% [8 427 ; 10 415] $/an
b : 24 396    IC95% [20 891 ; 27 901] $
```

La pente exclut 0 → **significativité statistique confirmée**.

---

## Prédiction avec intervalle de prédiction

```python
def predict(years_experience, confidence=0.95):
    yhat  = w_orig * years_experience + b_orig
    t_val = stats.t.ppf(1 - alpha/2, df=df)
    se_ip = s * sqrt(1 + 1/n + (x - x_bar)² / Sxx)
    margin = t_val * se_ip
    return yhat, yhat - margin, yhat + margin
```

Exemples :

| Expérience | Prédiction | IP 95% |
|------------|-----------|--------|
| 3.8 ans | 60 154 $ | [45 892 $ ; 74 416 $] |
| 6.0 ans | 80 920 $ | [66 918 $ ; 94 922 $] |
| 9.5 ans | 113 893 $ | [97 610 $ ; 130 176 $] |

---

## Limites

- **n = 30 :** dataset très réduit — les métriques hold-out restent sensibles à la composition du split. Le k-fold atténue ce biais.
- **Variable unique :** le salaire dépend d'autres facteurs non capturés (secteur, diplôme, localisation, poste).
- **Linéarité supposée :** la relation est validée sur 1.2–10.6 ans ; l'extrapolation est déconseillée hors de cette plage.
- **Modèle univarié :** une extension multivariée (régression multiple) améliorerait significativement le pouvoir explicatif.

---

## Installation & exécution

```bash
# Cloner le dépôt
git clone https://github.com/delsDin/SalaryMapper.git
cd SalaryMapper

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Lancer le notebook
jupyter notebook main.ipynb
```

**Dépendances :**

| Package | Rôle |
|---------|------|
| `numpy` | Calculs vectorisés (gradient, normalisation) |
| `pandas` | Chargement et exploration du dataset |
| `matplotlib` / `seaborn` | Visualisations |
| `scipy.stats` | Tests statistiques (Shapiro-Wilk, Student, probplot) |
| `scikit-learn` | Référence de validation uniquement |

---

## Contenu du notebook

| Section | Description |
|---------|-------------|
| 1. Imports | Configuration de l'environnement |
| 2. Chargement & exploration | Shape, valeurs manquantes, stats descriptives |
| 3. Visualisation initiale | Scatter, histogrammes des distributions |
| 4. Split & normalisation z-score | Train/test + prévention du data leakage |
| 5. Implémentation from scratch | Fonction de coût, descente de gradient |
| 6. Entraînement | Convergence, dénormalisation des paramètres |
| 7. Évaluation | R², RMSE, MAE sur train et test |
| 8. Courbe d'apprentissage | Visualisation de la convergence du coût |
| 9. Droite de régression | Overlay train/test/droite |
| 10. Validation croisée k-fold | 5 plis, estimation robuste |
| 11. Validation vs sklearn | Comparaison paramétrique |
| 12. Analyse des résidus | 6 graphiques diagnostiques |
| 13. Intervalles de confiance | IC 95% sur w et b (Student) |
| 14. Bande IC + IP | Visualisation graphique de l'incertitude |
| 15. Synthèse & limites | Équation finale, tableau de bord, limitations |

---

## Auteur

**Dels Dinla Marcel** — Licence 1 Informatique · IFRI, Bénin
Certifications : Python & Data Science (OpenClassrooms), Regression Linéaire (Machine Learning | IBM - Coursera)

Projets connexes : [Delsat](https://github.com/delsDin/DelsAt) · [Delsio](https://github.com/delsDin/Delsio)

---

*SalaryMapper — Régression linéaire from scratch · Gradient Descent · Normalisation z-score · k-Fold CV*

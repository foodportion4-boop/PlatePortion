"""Calibrate the geometric branch: volume -> mass -> nutrition.

The geometry gives volume in cm3. Turning that into grams needs a density, and
grams into nutrition needs a composition. Neither is recoverable from depth
alone, so both are fitted on the **training split** and then frozen:

    mass_g      = rho * volume_cm3
    calories    = mass_g * kcal_per_g
    fat/carb/protein = mass_g * (grams of macro per gram of food)

``rho`` is fitted by a robust regression through the origin (Theil-Sen style
median-of-ratios), not least squares: the volume estimates have a heavy right
tail from plate-fit failures, and OLS chases them.

:class:`Calibration` is deliberately the *simplest honest* mapping: one global
density, one global composition. :class:`IngredientCalibration` conditions both
on which ingredients are present.

**Measured, against the intuition:** conditioning the *density* on ingredients
is worth +0.2 points of mean PMAE; conditioning the *composition* is worth
+10.6. Volume -> mass is limited by how well the geometry measures volume
(r=0.673 against true mass), not by knowing the right density, so density
refinements have nowhere to contribute. Reproduce with
``scripts/ablate_calibration.py``. This is why no literature density table lives
here -- it would invite work the evidence says not to do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from n5k.data.metadata import TARGETS, Dish

@dataclass
class Calibration:
    """Frozen constants fitted on the training split."""

    density_g_per_cm3: float
    kcal_per_g: float
    fat_per_g: float
    carb_per_g: float
    protein_per_g: float
    n_train: int
    notes: str = ""

    def predict(self, volume_cm3: float) -> dict[str, float]:
        """Map a volume to the five official targets."""
        mass = max(volume_cm3, 0.0) * self.density_g_per_cm3
        return {
            "calories": mass * self.kcal_per_g,
            "mass": mass,
            "fat": mass * self.fat_per_g,
            "carb": mass * self.carb_per_g,
            "protein": mass * self.protein_per_g,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))


def fit_calibration(
    volumes: dict[str, float],
    dishes: dict[str, Dish],
    train_ids: list[str],
) -> Calibration:
    """Fit density and composition constants on the training split.

    Args:
        volumes: ``{dish_id: volume_cm3}`` from the geometric pipeline.
        dishes: parsed metadata.
        train_ids: **training** dish IDs only. Passing test IDs here leaks.
    """
    v, m = [], []
    kcal, fat, carb, prot = [], [], [], []
    for did in train_ids:
        if did not in volumes or did not in dishes:
            continue
        vol, dish = volumes[did], dishes[did]
        if vol <= 0 or dish.mass <= 0:
            continue
        v.append(vol)
        m.append(dish.mass)
        kcal.append(dish.calories / dish.mass)
        fat.append(dish.fat / dish.mass)
        carb.append(dish.carb / dish.mass)
        prot.append(dish.protein / dish.mass)

    if len(v) < 20:
        raise ValueError(
            f"only {len(v)} usable training dishes for calibration; "
            f"download more imagery before fitting"
        )

    v_arr, m_arr = np.asarray(v), np.asarray(m)
    # Median of per-dish ratios: robust to the heavy right tail of volume
    # outliers that OLS through the origin would chase.
    density = float(np.median(m_arr / v_arr))

    return Calibration(
        density_g_per_cm3=density,
        kcal_per_g=float(np.median(kcal)),
        fat_per_g=float(np.median(fat)),
        carb_per_g=float(np.median(carb)),
        protein_per_g=float(np.median(prot)),
        n_train=len(v),
        notes=(
            "Single global density fitted by median-of-ratios on the official "
            "depth train split. Composition constants are median per-gram "
            "macros over the same dishes."
        ),
    )


#: Ingredients below this train frequency are pooled into the intercept rather
#: than given their own coefficient -- a density fitted from three dishes is
#: noise, and it would fire on test dishes it cannot support.
MIN_INGREDIENT_COUNT = 20


@dataclass
class IngredientCalibration:
    """Volume -> nutrition using the *set* of ingredients on the plate.

    The single-density model in :class:`Calibration` cannot tell a salad from a
    curry of equal volume, and the measured results show that -- not the
    geometry -- is where the geometric branch's error lives. This model
    conditions the volume->mass and mass->nutrition constants on which
    ingredients are present.

    **Identity only, never quantities.** The feature vector is a binary
    indicator over ingredient names. Ground-truth *grams* are used to fit on
    train and never at inference; feeding them in at test time would be
    equivalent to reading out the answer, since dish mass is their sum.

    Using true ingredient identity at test time makes this an **oracle**: it
    upper-bounds what a perfect ingredient classifier would buy, and is not a
    deployable predictor on its own. Report it as such.
    """

    ingredients: list[str]
    #: (n_targets, n_ingredients + 1) ridge weights; last column is the intercept.
    weights: list[list[float]]
    fallback: "Calibration"
    n_train: int
    notes: str = ""

    #: Quantities predicted per gram of food, in order.
    RATIOS = ("density", "kcal_per_g", "fat_per_g", "carb_per_g", "protein_per_g")

    def _features(self, ingredient_names) -> np.ndarray:
        idx = {name: i for i, name in enumerate(self.ingredients)}
        x = np.zeros(len(self.ingredients) + 1, dtype=np.float64)
        x[-1] = 1.0
        hits = 0
        for name in ingredient_names:
            if name in idx:
                x[idx[name]] = 1.0
                hits += 1
        if hits:
            # Normalise so a 10-ingredient dish is not summed to 10x a
            # 1-ingredient dish; we want an average property, not a total.
            x[:-1] /= hits
        return x

    def predict(self, volume_cm3: float, ingredient_names) -> dict[str, float]:
        volume_cm3 = max(volume_cm3, 0.0)
        known = [n for n in ingredient_names if n in set(self.ingredients)]
        if not known:
            return self.fallback.predict(volume_cm3)

        x = self._features(known)
        vals = np.asarray(self.weights, dtype=np.float64) @ x
        density, kcal, fat, carb, prot = (float(v) for v in vals)

        # Ridge can extrapolate to nonsense on unusual ingredient combinations.
        # Clamp to physically sane bands rather than emitting negative mass.
        density = float(np.clip(density, 0.05, 1.6))
        mass = volume_cm3 * density
        return {
            "calories": mass * max(kcal, 0.0),
            "mass": mass,
            "fat": mass * max(fat, 0.0),
            "carb": mass * max(carb, 0.0),
            "protein": mass * max(prot, 0.0),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = asdict(self)
        blob["fallback"] = asdict(self.fallback)
        Path(path).write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IngredientCalibration":
        blob = json.loads(Path(path).read_text())
        blob["fallback"] = Calibration(**blob["fallback"])
        return cls(**blob)

    def density_table(self) -> dict[str, float]:
        """Per-ingredient density implied by the fitted weights, for inspection.

        Reads off the model's prediction for a hypothetical single-ingredient
        dish.

        **These are model coefficients, not measurements -- do not cite them.**
        Ingredients co-occur heavily (olive oil appears in 1,338 of 2,757 train
        dishes), so the coefficients are badly collinear and many saturate the
        clip bounds: the fit puts rosemary and soy sauce at 1.60 g/cm3, which is
        nonsense for trace seasonings. The ablation shows density conditioning
        contributes ~0.2 points regardless, so this exists to diagnose the fit,
        not to report densities.
        """
        w = np.asarray(self.weights, dtype=np.float64)
        out = {}
        for i, name in enumerate(self.ingredients):
            x = np.zeros(len(self.ingredients) + 1)
            x[i], x[-1] = 1.0, 1.0
            out[name] = float(np.clip((w @ x)[0], 0.05, 1.6))
        return out


def fit_ingredient_calibration(
    volumes: dict[str, float],
    dishes: dict[str, Dish],
    train_ids: list[str],
    alpha: float = 1.0,
    min_count: int | None = None,
) -> IngredientCalibration:
    """Ridge-fit per-ingredient density and composition on the training split.

    Targets are the *observed* per-dish ratios -- effective density ``M/V`` and
    per-gram macros -- regressed onto normalised ingredient indicators. Fitting
    the observed ratio directly (rather than solving a physical mixture model
    for each ingredient's density) keeps the objective aligned with the quantity
    actually used at inference, and degrades gracefully when the volume estimate
    is imperfect, which it is.
    """
    if min_count is None:
        # Scale with the training set so smoke runs on a few dozen dishes still
        # fit something, while a full run keeps the stricter threshold.
        min_count = max(3, min(MIN_INGREDIENT_COUNT, len(train_ids) // 100))

    counts: dict[str, int] = {}
    for did in train_ids:
        if did in dishes:
            for ing in dishes[did].ingredients:
                counts[ing.name] = counts.get(ing.name, 0) + 1
    names = sorted(n for n, c in counts.items() if c >= min_count)
    if not names:
        raise ValueError(f"no ingredient reaches min_count={min_count}")

    fallback = fit_calibration(volumes, dishes, train_ids)

    rows, targets = [], []
    idx = {n: i for i, n in enumerate(names)}
    for did in train_ids:
        if did not in volumes or did not in dishes:
            continue
        vol, dish = volumes[did], dishes[did]
        if vol <= 0 or dish.mass <= 0:
            continue
        present = [ing.name for ing in dish.ingredients if ing.name in idx]
        if not present:
            continue
        x = np.zeros(len(names) + 1)
        for n in present:
            x[idx[n]] = 1.0
        x[:-1] /= len(present)
        x[-1] = 1.0
        rows.append(x)
        targets.append([
            dish.mass / vol,
            dish.calories / dish.mass,
            dish.fat / dish.mass,
            dish.carb / dish.mass,
            dish.protein / dish.mass,
        ])

    if len(rows) < 20:
        raise ValueError(
            f"only {len(rows)} usable training dishes for the ingredient fit "
            f"(need >=20). Download more imagery, or drop --limit."
        )
    if len(rows) < len(names):
        # Under-determined: ridge will still solve, but the coefficients are
        # barely constrained. Say so rather than presenting them as measured.
        print(
            f"  warning: fitting {len(names)} ingredient coefficients from only "
            f"{len(rows)} dishes -- densities are weakly identified"
        )

    A = np.asarray(rows)
    Y = np.asarray(targets)
    # Winsorise the density target: M/V blows up when the volume estimate
    # collapses, and a handful of such dishes would dominate a squared loss.
    lo, hi = np.percentile(Y[:, 0], [1, 99])
    Y[:, 0] = np.clip(Y[:, 0], lo, hi)

    # Ridge, with the intercept column left unpenalised.
    P = np.eye(A.shape[1]) * alpha
    P[-1, -1] = 0.0
    W = np.linalg.solve(A.T @ A + P, A.T @ Y).T

    return IngredientCalibration(
        ingredients=names,
        weights=W.tolist(),
        fallback=fallback,
        n_train=len(rows),
        notes=(
            f"Ridge (alpha={alpha}) on normalised ingredient indicators, fitted "
            f"on the official depth train split. Ingredients with fewer than "
            f"{min_count} train occurrences are pooled into the intercept. "
            f"ORACLE: consumes ground-truth ingredient identity at inference, "
            f"so it upper-bounds a perfect ingredient classifier rather than "
            f"being deployable. Quantities are never used at inference."
        ),
    )


__all__ = [
    "Calibration",
    "IngredientCalibration",
    "fit_calibration",
    "fit_ingredient_calibration",
    "TARGETS",
]

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class AVOType(Enum):
    CLASS_I = "I"
    CLASS_II = "II"
    CLASS_III = "III"
    CLASS_IV = "IV"
    UNKNOWN = "Unknown"


@dataclass
class AVOClassification:
    avo_type: AVOType
    intercept_A: float
    gradient_B: float
    ratio_AB: float
    critical_angle: Optional[float]
    confidence: float


def classify_avo(
    angles: np.ndarray,
    amplitudes: np.ndarray,
    min_r_squared: float = 0.7
) -> AVOClassification:
    sin2_theta = np.sin(np.deg2rad(angles))**2
    X = sin2_theta.reshape(-1, 1)
    y = amplitudes

    model = LinearRegression()
    model.fit(X, y)

    A = model.intercept_
    B = model.coef_[0]
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    if r2 < min_r_squared:
        return AVOClassification(
            avo_type=AVOType.UNKNOWN,
            intercept_A=A,
            gradient_B=B,
            ratio_AB=A / (B + 1e-10),
            critical_angle=None,
            confidence=r2
        )

    ratio_AB = A / (B + 1e-10)
    critical_angle = None

    if A > 0 and B < 0 and abs(A) > abs(B):
        R_30deg = A + B * np.sin(np.deg2rad(30))**2
        if R_30deg > 0:
            avo_type = AVOType.CLASS_I
        else:
            avo_type = AVOType.CLASS_II
    elif B < 0 and abs(B) > abs(A):
        avo_type = AVOType.CLASS_II
        if -A / B > 0 and -A / B <= 1:
            sin2_crit = -A / B
            critical_angle = np.rad2deg(np.arcsin(np.sqrt(sin2_crit)))
        else:
            critical_angle = None
    elif A < 0 and B < 0:
        avo_type = AVOType.CLASS_III
    elif A < 0 and B > 0:
        avo_type = AVOType.CLASS_IV
    else:
        avo_type = AVOType.UNKNOWN

    return AVOClassification(
        avo_type=avo_type,
        intercept_A=A,
        gradient_B=B,
        ratio_AB=ratio_AB,
        critical_angle=critical_angle,
        confidence=r2
    )

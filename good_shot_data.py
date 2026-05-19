"""
good_shot_data.py — Krepšinio metimo biomechaniniai etaloniniai intervalai.

"""

from dataclasses import dataclass, field
from typing import Union

# ── Data structure ─────────────────────────────────────────────────────────────
@dataclass
class AngleRange:
    name: str
    min_deg: float
    max_deg: float
    unit: str = 'deg'
    description: str = ''

    def contains(self, value: float) -> bool:
        return self.min_deg <= value <= self.max_deg

    def deviation(self, value: float) -> float:
        if value < self.min_deg:
            return self.min_deg - value
        if value > self.max_deg:
            return value - self.max_deg
        return 0.0

    def __repr__(self):
        return f'{self.name}: {self.min_deg}–{self.max_deg}°'


# ── Reference ranges per phase ─────────────────────────────────────────────────

REFERENCE: dict[str, dict[str, AngleRange]] = {

    # ── Frame 1: Set Pose ──────────────────────────────────────────────
    'READY': {
        # Lower Body
        'knee': AngleRange(
            name='Knee flexion',
            min_deg=40, max_deg=70,
            description='Moderate knee bend in set stance'
        ),
        'hip': AngleRange(
            name='Hip flexion',
            min_deg=30, max_deg=50,
            description='Slight forward hip flexion'
        ),
        'ankle': AngleRange(
            name='Ankle angle',
            min_deg=65, max_deg=80,
            description='Dorsiflexion in set pose (90-25 to 90-10)'
        ),
        # Shooting Shoulder
        'shoulder_flexion': AngleRange(
            name='Shoulder flexion',
            min_deg=40, max_deg=70,
            description='Shoulder elevated to set position'
        ),
        # Shooting Elbow
        'elbow': AngleRange(
            name='Elbow angle',
            min_deg=70, max_deg=100,
            description='Flexion — elbow under ball'
        ),
        # Wrist
        'wrist': AngleRange(
            name='Wrist angle',
            min_deg=140, max_deg=160,
            description='Extension — cocked back at set'
        ),
        # Ball position
        '_ball_height': 'Waist–chin',
    },

    # ── Frame 2: Loaded Upward Pose ────────────────────────────────────
    'LOADING': {
        # Lower Body
        'knee': AngleRange(
            name='Knee flexion',
            min_deg=60, max_deg=90,
            description='Deeper knee bend during loading'
        ),
        'hip': AngleRange(
            name='Hip flexion',
            min_deg=40, max_deg=60,
            description='Hip loaded for upward drive'
        ),
        'ankle': AngleRange(
            name='Ankle angle',
            min_deg=85, max_deg=110,
            description='Dorsiflexion during loading (90-30 to 90-15)'
        ),
        # Shooting Shoulder
        'shoulder_flexion': AngleRange(
            name='Shoulder flexion',
            min_deg=70, max_deg=100,
            description='Shoulder rising with ball'
        ),
        # Shooting Elbow
        'elbow': AngleRange(
            name='Elbow angle',
            min_deg=85, max_deg=130,
            description='Flexion — elbow driving upward'
        ),
        # Wrist
        'wrist': AngleRange(
            name='Wrist angle',
            min_deg=120, max_deg=145,
            description='Extension — loaded for release'
        ),
        # Ball position
        '_ball_height': 'Eye level',
    },

    # ── Frame 3: Pre-Release Pose ──────────────────────────────────────
    'RELEASE': {
        # Lower Body
        'knee': AngleRange(
            name='Knee flexion',
            min_deg=140, max_deg=170,
            description='Near full extension at release'
        ),
        'hip': AngleRange(
            name='Hip flexion',
            min_deg=150, max_deg=180,
            description='Near full hip extension'
        ),
        'ankle': AngleRange(
            name='Ankle angle',
            min_deg=100, max_deg=130,
            description='Plantarflexion at toe-off'
        ),
        # Shooting Shoulder
        'shoulder_flexion': AngleRange(
            name='Shoulder flexion',
            min_deg=110, max_deg=140,
            description='High shoulder flexion at release'
        ),
        # Shooting Elbow
        'elbow': AngleRange(
            name='Elbow angle',
            min_deg=130, max_deg=180,
            description='Extension — fully extending at release'
        ),
        # Wrist
        'wrist': AngleRange(
            name='Wrist angle',
            min_deg=135, max_deg=180,
            description='Flexion — snapping through at release'
        ),
        # Ball position
        '_ball_height': 'Above eye',
    },
}


# ── Phase name aliases (maps process_frame phases to reference keys) ───────────
PHASE_ALIAS: dict[str, str] = {
    'READY': 'READY',
    'LOADING': 'LOADING',
    'RELEASE': 'RELEASE',
    None: 'READY',
}


# ── Helper functions ───────────────────────────────────────────────────────────
def get_phase_ref(phase: str) -> dict[str, Union[AngleRange, str]]:
    key = PHASE_ALIAS.get(phase, 'READY')
    return REFERENCE[key]


def check_angle(param_key: str, value: float, phase: str) -> tuple[bool, str]:
    ref_dict = get_phase_ref(phase)
    if param_key not in ref_dict or not isinstance(ref_dict[param_key], AngleRange):
        return True, 'NO_REF'

    r = ref_dict[param_key]
    if value < r.min_deg:
        return False, f'LOW (measured {value:.0f}°, expected ≥{r.min_deg}°)'
    if value > r.max_deg:
        return False, f'HIGH (measured {value:.0f}°, expected ≤{r.max_deg}°)'
    return True, 'OK'


def deviation_score(param_key: str, value: float, phase: str) -> float:
    ref_dict = get_phase_ref(phase)
    if param_key not in ref_dict or not isinstance(ref_dict[param_key], AngleRange):
        return 0.0
    r = ref_dict[param_key]
    dev = r.deviation(value)
    width = max(1.0, r.max_deg - r.min_deg)
    return dev / width


def summary(phase: str) -> str:
    ref_dict = get_phase_ref(phase)
    lines = [f'--- {phase} reference ranges ---']
    for key, val in ref_dict.items():
        if key.startswith('_'):
            lines.append(f'Ball position : {val}')
        else:
            lines.append(f' {val}')
    return '\n'.join(lines)


# ── Quick sanity check when run directly ──────────────────────────────────────
if __name__ == '__main__':
    for phase in ('READY', 'LOADING', 'RELEASE'):
        print(summary(phase))
        print()

    print('--- check_angle examples ---')
    print('knee=55 READY:', check_angle('knee', 55, 'READY'))
    print('knee=30 READY:', check_angle('knee', 30, 'READY'))
    print('elbow=160 RELEASE:', check_angle('elbow', 160, 'RELEASE'))
    print('elbow=120 RELEASE:', check_angle('elbow', 120, 'RELEASE'))
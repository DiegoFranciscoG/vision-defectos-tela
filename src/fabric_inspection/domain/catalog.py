"""Class catalog of MVTec AD carpet and its mapping to textrack defect codes."""

from dataclasses import dataclass

GOOD = "good"


@dataclass(frozen=True, slots=True)
class DefectClass:
    code: str
    name_es: str
    is_defect: bool
    is_hole: bool
    # Code of textrack's `defect_types` catalog; None when textrack has no equivalent.
    textrack_code: str | None


DEFECT_CLASSES: tuple[DefectClass, ...] = (
    DefectClass(GOOD, "Sin defecto", is_defect=False, is_hole=False, textrack_code=None),
    DefectClass("color", "Mancha o variación de color", True, False, "SHADE_VARIATION"),
    DefectClass("cut", "Corte en la superficie", True, False, None),
    DefectClass("hole", "Agujero", True, True, "FABRIC_HOLE"),
    DefectClass("metal_contamination", "Fragmento metálico", True, False, "BROKEN_NEEDLE"),
    DefectClass("thread", "Hilo suelto", True, False, "LOOSE_THREAD"),
)

CLASS_CODES: tuple[str, ...] = tuple(item.code for item in DEFECT_CLASSES)
DEFECT_CODES: tuple[str, ...] = tuple(item.code for item in DEFECT_CLASSES if item.is_defect)
TEXTRACK_CODES: dict[str, str | None] = {item.code: item.textrack_code for item in DEFECT_CLASSES}


def get_class(code: str) -> DefectClass:
    for item in DEFECT_CLASSES:
        if item.code == code:
            return item
    raise KeyError(f"Unknown defect class: {code}")

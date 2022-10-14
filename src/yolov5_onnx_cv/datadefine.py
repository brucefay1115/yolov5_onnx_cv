from dataclasses import dataclass


@dataclass
class ClassInfo:
    id: int
    name: str
    conf: float
    box: list
from __future__ import annotations

from typing import Type, Union

from audio_engine.core.operator import BaseOperator, ManifestOperator

OperatorType = Union[BaseOperator, ManifestOperator]


class OperatorRegistry:
    """Central registry for operator lookup by dotted name."""

    _operators: dict[str, Type[OperatorType]] = {}

    @classmethod
    def register(cls, operator_cls: Type[OperatorType]) -> Type[OperatorType]:
        instance = operator_cls()
        key = instance.full_name
        cls._operators[key] = operator_cls
        return operator_cls

    @classmethod
    def get(cls, name: str) -> OperatorType:
        if name not in cls._operators:
            available = ", ".join(sorted(cls._operators))
            raise KeyError(f"Operator '{name}' not registered. Available: {available}")
        return cls._operators[name]()

    @classmethod
    def list_operators(cls) -> list[str]:
        return sorted(cls._operators.keys())

    @classmethod
    def clear(cls) -> None:
        cls._operators.clear()


def register_operator(operator_cls: Type[OperatorType]) -> Type[OperatorType]:
    return OperatorRegistry.register(operator_cls)

import pydantic as pyd


class Parameter(pyd.BaseModel):
    type: str


class Function(pyd.BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class Prompts(pyd.BaseModel):
    prompt: str


class FunctionCall(pyd.BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]


def _matches(type_name: str, value: object) -> bool:
    """Check that a value fits a declared parameter type."""
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    return False


def validate_call(call: FunctionCall, func: Function) -> None:
    """Check a call against its function definition.

    Args:
        call: The generated call.
        func: The definition of the function it names.

    Raises:
        ValueError: If keys differ or a value has the wrong type.
    """
    if call.name != func.name:
        raise ValueError(f"Name mismatch: {call.name} vs {func.name}")
    if set(call.parameters) != set(func.parameters):
        raise ValueError(f"Parameter keys differ for {func.name}")
    for key, param in func.parameters.items():
        if not _matches(param.type, call.parameters[key]):
            raise ValueError(
                f"Parameter {key!r} is not of type {param.type!r}"
            )

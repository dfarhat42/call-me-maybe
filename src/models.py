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

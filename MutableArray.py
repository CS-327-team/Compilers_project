from dataclasses import dataclass
from typing import Union, Mapping, List


@dataclass
class NumLiteral:
    value: Union[int, float]

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'

@dataclass
class Index:
    array: 'AST'
    index: 'AST'

@dataclass
class Append:
    array: 'AST'
    value: 'AST'

@dataclass
class Pop:
    array: 'AST'

@dataclass
class Concat:
    left: 'AST'
    right: 'AST'

@dataclass
class Assign:
    array: 'AST'
    index: 'AST'
    value: 'AST'

@dataclass
class MutableArray:
    elements: List['AST']

AST = NumLiteral | BinOp | Index | Append | Pop | Concat | Assign | MutableArray
Value = Union[int, float, List['Value']]

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case MutableArray(elements):
            return [eval(element, environment) for element in elements]
        case Index(array, index):
            array_value = eval(array, environment)
            index_value = eval(index, environment)
            if not isinstance(array_value, list):
                raise InvalidProgram("Can only index into a list")
            if not isinstance(index_value, int):
                raise InvalidProgram("Index must be an integer")
            try:
                return array_value[index_value]
            except IndexError:
                raise InvalidProgram("Index out of range")
        case Append(array, value):
            array_value = eval(array, environment)
            value = eval(value, environment)
            if not isinstance(array_value, list):
                raise InvalidProgram("Can only append to a list")
            array_value.append(value)
            return array_value
        case Pop(array):
            array_value = eval(array, environment)
            if not isinstance(array_value, list):
                raise InvalidProgram("Can only pop from a list")
            try:
                return array_value.pop()
            except IndexError:
                raise InvalidProgram("Can't pop from an empty list")
        case Concat(left, right):
            left_value = eval(left, environment)
            right_value = eval(right, environment)
            if not isinstance(left_value, list) or not isinstance(right_value, list):
                raise InvalidProgram("Can only concatenate two lists")
            return left_value + right_value
        case Assign(array, index, value):
            array_value = eval(array, environment)
            index_value = eval(index, environment)
            value = eval(value, environment)
            if not isinstance(array_value, list):
                raise InvalidProgram("Can only assign to a list")
            if not isinstance(index_value, int):
                raise InvalidProgram("Index must be an integer")
            try:
                array_value[index_value] = value
            except IndexError:
                raise InvalidProgram("Index out of range")
            return array_value
    raise InvalidProgram("Invalid program")

def test_eval():
    e1 = NumLiteral(1)
    e2 = NumLiteral(2)
    e3 = NumLiteral(3)
    e4 = MutableArray([e1, e2])
    e5 = Append(e4, e3)
    e6 = Index(e4, NumLiteral(0))
    e7 = Pop(e5)
    e8 = Concat(Concat(e4, e5), MutableArray([]))
    e9 = Assign(e4, NumLiteral(0), NumLiteral(0))


    assert eval(e4) == [1, 2]
    assert eval(e5) == [1, 2, 3]
    assert eval(e6) == 1
    assert eval(Index(e5, NumLiteral(2))) == 3
    assert eval(e7) == 3
    assert eval(e8) == [1, 2, 1, 2, 3]
    assert eval(e9) == [0, 2]


test_eval()
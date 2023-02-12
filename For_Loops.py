from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping

@dataclass
class NumLiteral:
    value: Fraction
    def __init__(self, *args):
        self.value = Fraction(*args)

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'

@dataclass
class Variable:
    name: str

@dataclass
class Print:
    def __init__(self, exp: "AST"):
        self.exp = exp

@dataclass
class ForLoop:
    var_name: str
    start: Union[NumLiteral, Variable]
    end: Union[NumLiteral, Variable]
    body: 'AST'

@dataclass
class While:
    condition: 'AST'
    body: 'AST'

AST = NumLiteral | BinOp | Variable | ForLoop | While

Value = Fraction

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            if name in environment:
                return environment[name]
            raise InvalidProgram()
        case BinOp("+", left, right):
            return eval(left, environment) + eval(right, environment)
        case BinOp("-", left, right):
            return eval(left, environment) - eval(right, environment)
        case BinOp("*", left, right):
            return eval(left, environment) * eval(right, environment)
        case BinOp("/", left, right):
            return eval(left, environment) / eval(right, environment)
        case BinOp(">", left, right):
            return eval(left, environment) > eval(right, environment)
        case BinOp("<", left, right):
            return eval(left, environment) < eval(right, environment)
        case BinOp("==", left, right):
            return eval(left, environment) == eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)        
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)        
        case ForLoop(var_name, start, end, body):
            start_value = eval(start, environment)
            end_value = eval(end, environment)
            if start_value >= end_value:
                raise InvalidProgram("Start value must be less than end value")
            result = 0
            for i in range(start_value, end_value + 1):
                result = eval(body, environment | {var_name: i})
            return result
        case While(condition, body):
            while eval(condition, environment) != 0:
                eval(body, environment)
            return None
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value
    raise InvalidProgram()

def test_for_loop():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("+", Variable("i"), e1)
    e4 = BinOp("<=", Variable("i"), e2)
    e5 = BinOp("*", Variable("result"), Variable("i"))
    e6 = ForLoop(Variable("i"), e1, e2, BinOp("=", Variable("result"), e5))
    environment = {"result": NumLiteral(1)}
    eval(e6, environment)
    assert environment["result"] == Fraction(120)

def test_while_loop():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("<=", Variable("i"), e2)
    e4 = BinOp("+", Variable("i"), e1)
    e5 = BinOp("*", Variable("result"), Variable("i"))
    e6 = BinOp("=", Variable("i"), e4)
    e7 = BinOp("=", Variable("result"), e5)
    e8 = While(e3, [e7, e6])
    environment = {"i": e1, "result": NumLiteral(1)}
    eval(e8, environment)
    assert environment["result"] == Fraction(120)
    print(environment["result"])

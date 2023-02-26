from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Union, Mapping

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

# @dataclass(unsafe_hash=True)
# class Variable:
#     name: str = field(hash=True)

#     def __hash__(self):
#         return hash(self.name)
@dataclass
class Put:
    var: 'AST'
    e1: 'AST'

@dataclass
class Get:
    var: 'AST'

@dataclass(unsafe_hash=True)
class Variable:
    name: str

    # def eval(self, environment: Environment) -> Union[int, Fraction]:
    #     if self.name in environment:
    #         return environment[self.name]
    #     raise InvalidProgram()

@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

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

AST = NumLiteral | BinOp | Variable | Let | Put | Get | ForLoop | While

Value = Fraction | NumLiteral

class InvalidProgram(Exception):
    pass

class Environment:
    envs: List

    def __init__(self) -> None:
        self.envs = [{}]
    
    def enter_scope(self):
        self.envs.append({})

    def exit_scope(self):
        assert self.envs
        self.envs.pop()

    def add(self, name, value):
        assert name not in self.envs[-1]
        self.envs[-1][name] = value

    def get(self, name):
        for env in reversed(self.envs):
            if name in env:
                return env[name]
        raise KeyError()

    def update(self, name, value):
        for env in reversed(self.envs):
            if name in env:
                env[name] = value
                return
        raise KeyError()
    
def eval(program: AST, environment: Environment = None) -> Value:
    if environment is None:
        environment = Environment()

    def eval_(program):
        return eval(program, environment)

    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            return environment.get(name)
        case Let(Variable(name), e1, e2):
            v1 = eval_(e1)
            environment.enter_scope()
            environment.add(name, v1)
            v2 = eval_(e2)
            environment.exit_scope()
            return v2
        case BinOp("+", left, right):
            return eval_(left) + eval_(right)
        case BinOp("-", left, right):
            return eval_(left) - eval_(right)
        case BinOp("*", left, right):
            return eval_(left) * eval_(right)
        case BinOp("/", left, right):
            return eval_(left) / eval_(right)
        case BinOp(">", left, right):
            return eval_(left) > eval_(right)
        case BinOp("<", left, right):
            return eval_(left) < eval_(right)
        case BinOp("==", left, right):
            return eval_(left) == eval_(right)
        case BinOp("<=", left, right):
            return eval_(left) <= eval_(right)        
        case BinOp(">=", left, right):
            return eval_(left) >= eval_(right)        
        case Put(Variable(name), e):
            try:
                environment.add(name, eval_(e))
            except:
                environment.update(name, eval_(e))
            return environment.get(name)
        case Get(Variable(name)):
            return environment.get(name)
        case ForLoop(var_name, start, end, body):
            start_value = int(eval_(start))
            # print(type(start_value))
            end_value = int(eval_(end))
            if start_value >= end_value:
                raise InvalidProgram("Start value must be less than end value")
            result = 0
            environment.add(var_name, 0)
            for i in range(start_value, end_value + 1):
                environment.update(var_name, i)
                environment.update(result, eval_(body))
                print(result)
            return result
        case While(condition, body):
            while eval_(condition) != 0:
                eval_(body)
            return None
        case Print(exp):
            value = eval_(exp)
            print(value)
            return value
    raise InvalidProgram()

def test_for_loop():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("+", Variable("i"), e1)
    e4 = BinOp("<=", Variable("i"), e2)
    e5 = BinOp("*", Variable("result"), Variable("i"))
    result = Variable("result")
    i = Variable("i")
    environment = Environment()
    environment.enter_scope()
    eval(Let(i, NumLiteral(1)), environment)
    eval(Let(result, NumLiteral(1)), environment)
    e6 = ForLoop(Variable("i"), e1, e2, Put(result, e5))
    eval(e6, environment)
    assert environment["result"] == NumLiteral(120)

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
    assert environment["result"] == NumLiteral(120)

test_for_loop()
# test_while_loop()
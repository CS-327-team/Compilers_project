from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, List


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

# Implementing let
@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

@dataclass
class ParallelLet:
    vars: List[Variable]
    exprs: List['AST']
    body: 'AST'


AST = NumLiteral | BinOp | Variable | Let | ParallelLet
Value = Fraction

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value # Returns the value of the NumLiteral
        case Variable(name):
            if name in environment:
                return environment[name] # Returns the value of the variable if it exists in the environment
            raise InvalidProgram() # Raises an error if the variable is not in the environment
        case BinOp("+", left, right):
            return eval(left, environment) + eval(right, environment) 
        case BinOp("-", left, right):
            return eval(left, environment) - eval(right, environment) 
        case BinOp("*", left, right):
            return eval(left, environment) * eval(right, environment) 
        case BinOp("/", left, right):
            return eval(left, environment) / eval(right, environment) 
        # Adding case for let
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment) # Evaluates the first expression
            return eval(e2, environment | { name: v1 }) # Evaluates the second expression with the first expression's result bound to the variable
        case ParallelLet(vars, exprs, body):
            if len(vars) != len(exprs):
                raise InvalidProgram()
            new_env = environment.copy()
            for var, expr in zip(vars, exprs):
                new_env[var.name] = eval(expr, environment)
            return eval(body, new_env)
    raise InvalidProgram() # Raises an error if the program is invalid


def test_eval():
    e1 = NumLiteral(4)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp("-", e1, e6)
    #print(f"e7: {e7}") 
    #print(f"eval(e7): {eval(e7)}")  
    assert eval(e7) == Fraction(4, 5)


def test_let_eval():
    a = Variable("a")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, a)
    e = Let(a, e1, e2)
    assert eval(e) == 10
    e = Let(a, e1, Let(a, e2, e2))
    assert eval(e) == 20

#test_eval()
test_let_eval()

def test_parallellet_eval():
    a = Variable("a")
    b = Variable("b")
    e1 = NumLiteral(5)
    e2 = NumLiteral(6)
    e3 = BinOp("+", a, b)
    e4 = BinOp("*", a, b)
    e = ParallelLet([a, b], [e1, e2], BinOp("+", e3, e4))
    assert eval(e) == 41

test_parallellet_eval()

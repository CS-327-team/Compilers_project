from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping
# adding libraries to test the print statement
from contextlib import redirect_stdout
from io import StringIO

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

# Implementing If-Else statement
@dataclass
class If:
  cond : 'AST'
  true_branch : 'AST'
  false_branch : 'AST'

# Implementing the Boolean Type
@dataclass
class BoolLiteral:
    value : bool
    def __init__(self, value:bool):
        self.value = value

# implementing the print function
@dataclass
class Print:
    def __init__(self, exp : 'AST'):
        self.exp = exp
        
@dataclass
class Function:
    params: List[str]
    body: 'AST'

AST = NumLiteral | BinOp | Variable | If | BoolLiteral | Print

Value = [Fraction, bool]       # updated Value, for BoolLiteral

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
        # adding case for if-else 
        case If(cond, true_branch, false_branch):
          if eval(cond, environment):
            return eval(true_branch, environment)
          else:
            return eval(false_branch, environment)
        # adding BoolLiteral case
        case BoolLiteral(value):
            return value
        # comparison operators 
        case BinOp("<", left, right):
            return eval(left, environment) < eval(right, environment)
        case BinOp(">", left, right):
            return eval(left, environment) > eval(right, environment)
        case BinOp("==", left, right):
            return eval(left, environment) == eval(right, environment)
        case BinOp("!=", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)
        # adding case for print statement
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value
        # adding case for function 
        case Function(params, body):
            def func(*args):
                if len(args) != len(params):
                    raise InvalidProgram("Incorrect number of arguments")
                local_env = dict(environment)
                for name, value in zip(params, args):
                    local_env[name] = value
                return eval(body, local_env)
            return func
    raise InvalidProgram()

def test_if_else_eval():
  e1 = NumLiteral(2)
  e2 = NumLiteral(7)
  e3 = NumLiteral(9)
  e4 = BinOp(">", e2, e3)
  e5 = BinOp("+", e2, e3)
  e = If(e4, e2, e5) 
  assert eval(e) == 9

def test_bool_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(3)
    e4 = NumLiteral(4)
    e5 = BoolLiteral(True)
    e6 = BoolLiteral(False)
    e7 = BinOp("<", e2, e1)
    e8 = BinOp(">", e2, e1)  
    e9 = BinOp("==", e7, e6)
    e10 = BinOp("!=", e8, e5)
    e11 = BinOp("<=", e2, e3)
    e12 = BinOp(">=", e1, e4)  
    assert eval(e7) == False
    assert eval(e8) == True
    assert eval(e9) == True
    assert eval(e10) == True
    assert eval(e11) == False
    assert eval(e12) == False

# testing the print statement
def test_print_eval():
    e1 = NumLiteral(2)
    to_print = Print(e1)
    
# testing the Function class
def test_function_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(3)
    e3 = BinOp("*", Variable("x"), Variable("y"))
    f = Function(["x", "y"], e3)
    assert eval(f, {"x": e1, "y": e2}) == 6

    temp = StringIO()
    with redirect_stdout(temp):
        eval(to_print)
    ans = temp.getvalue
    assert ans == "2\n"

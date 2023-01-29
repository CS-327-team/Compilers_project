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
    left: "AST"
    right: "AST"


@dataclass
class Variable:
    name: str

    def slicing(self, start_index: NumLiteral, end_index: NumLiteral):

        if start_index > len(self.name) - 1:
            raise IndexError()
        elif end_index <= start_index:
            raise IndexError()
        elif end_index > len(self.name):
            raise IndexError()
        else:
            string_slice = self.name[start_index:end_index]
            return Variable(string_slice)


@dataclass
class If:
    cond: "AST"
    true_branch: "AST"
    false_branch: "AST"


# Implementing the BoolLiteral Type
@dataclass
class BoolLiteral:
    value: bool

    def __init__(self, val:bool):
        self.value = val


# implementing the print function
@dataclass
class Print:
    def __init__(self, exp: "AST"):
        self.exp = exp


@dataclass
class Let:
    var: "AST"
    e1: "AST"
    e2: "AST"


AST = NumLiteral | BinOp | Variable | Let | BoolLiteral

Value = Fraction


class InvalidProgram(Exception):
    pass


def typeof(s: AST):
    match s:
        case Variable(name):
            return "type string"
        case NumLiteral(value):
            return "type Fraction"
        case BoolLiteral(value):
            return "type boolean"
        case BinOp("+", left, right):
            if typeof(left) != typeof(right):
                return TypeError()
            elif typeof(left) == "type boolean":
                return TypeError()
            else:
                return typeof(left)
        case BinOp("-", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp("*", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp("/", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp(">", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("<", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("==", left, right):
            if typeof(left) == typeof(right):
                return "type boolean"
            else:
                raise TypeError()
        case BinOp(">=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("<=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("!=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("=", left, right):
            if typeof(left) == typeof(right):
                return
            else:
                raise TypeError()
    raise TypeError()


def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    typeof(program)
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            return name
        case BoolLiteral(value):
            return value
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment)
            return eval(e2, environment | {name: v1})
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
        case If(cond, true_branch, false_branch):
            if eval(cond, environment):
                return eval(true_branch, environment)
            else:
                return eval(false_branch, environment)
        case BinOp("!=", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)
        case BinOp("=", left, right):
            right_eval = eval(right)
            match left:
                case Variable(name):
                    left.name = right_eval
                    return
                case BoolLiteral(value):
                    left.value = right_eval
                    return
                case NumLiteral(value):
                    left.value = right_eval
                    return
            raise InvalidProgram()
        # adding case for print statement
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value

    raise InvalidProgram()


def boolify(s: AST):
    e = eval(s)
    if typeof(e) == "type Fraction":
        return BoolLiteral(True) if e > 0 else BoolLiteral(False)
    elif typeof(e) == "type string":
        return BoolLiteral(True) if len(e) > 0 else BoolLiteral(False)
    else:
        return s


def test_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp("*", e1, e6)
    assert eval(e7) == Fraction(32, 5)


def test_let_eval():
    a = Variable("a")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, a)
    e = Let(a, e1, e2)
    assert eval(e) == 10
    e = Let(a, e1, Let(a, e2, e2))
    assert eval(e) == 20
    e = Let(a, e1, BinOp("+", a, Let(a, e2, e2)))
    assert eval(e) == 25
    e = Let(a, e1, BinOp("+", Let(a, e2, e2), a))
    assert eval(e) == 25
    e3 = NumLiteral(6)
    e = BinOp("+", Let(a, e1, e2), Let(a, e3, e2))
    assert eval(e) == 22

print(eval(BinOp(">",NumLiteral(3),NumLiteral(5))))

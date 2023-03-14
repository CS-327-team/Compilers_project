from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, Literal

digit_list = "1234567890"
alphabet_list = "ABCDEFGHIJKLOMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
variable_list = []


@dataclass
class EndOfStream(Exception):
    pass


@dataclass
class Stream:
    string: str
    position: int

    def next_char(self):
        if self.position >= len(self.string):
            raise EndOfStream()
        else:
            self.position += 1
            return self.string[self.position - 1]

    def unget(self):
        assert self.position > 0
        self.position -= 1


@dataclass
class Num:
    n: int


@dataclass
class Bool:
    b: bool


@dataclass
class Keyword:
    word: str


@dataclass
class Identifier:
    word: str


@dataclass
class Operator:
    operator: str


@dataclass
class Paranthesis:
    paran: str


@dataclass
class Delimiter:
    delim: str


Token = Num | Bool | Keyword | Identifier | Operator | Paranthesis | Delimiter


@dataclass
class EndOfTokens(Exception):
    pass


def convert_to_token(c):
    if c == "True" or c == "False":
        return Bool(c)
    elif c in ["=", ">", "<", "+", "-", "*", "/", "!=", "<=", ">=", "^", "%"]:
        return Operator(c)
    elif c in "if then else end while do done".split():
        return Keyword(c)
    return Identifier(c)


@dataclass
class EndOfToken(Exception):
    pass


@dataclass
class TokenError(Exception):
    pass


operations = ["=", ">", "<", "+", "-", "*", "/", "!=", "<=", ">=", "%", "^"]
keywords = "if then else end while do done".split()
delimiters = ['"', ";"]


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = -1
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def backtrack(self):
        assert self.pos > 0
        self.pos -= 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def tokenize(self):
        tokens = []
        while self.current_char != None:
            match self.current_char:
                case op if op in ["+", "-", "*", "/", "%", "^"]:
                    tokens.append(Operator(op))
                    self.advance()
                case op if op in ["!", "=", ">", "<"]:
                    self.advance()
                    if self.current_char == "=":
                        op += self.current_char
                        self.advance()
                    tokens.append(Operator(op))
                case space if space in " \t\n":
                    self.advance()
                case num if num in digit_list:
                    num_temp = ""
                    while self.current_char != None and self.current_char in digit_list:
                        num_temp += self.current_char
                        self.advance()
                    tokens.append(Num(int(num_temp)))
                case s if s in alphabet_list:
                    temp_str = ""
                    while (
                        self.current_char != None
                        and self.current_char in alphabet_list + digit_list
                    ):
                        temp_str += self.current_char
                        self.advance()
                    tokens.append(
                        Keyword(temp_str)
                    ) if temp_str in keywords else tokens.append(
                        Bool(temp_str)
                    ) if temp_str in "True False".split() else tokens.append(
                        Identifier(temp_str)
                    )
                case s if s in "()":
                    tokens.append(Paranthesis(s))
                    self.advance()
                case s if s in delimiters:
                    tokens.append(Delimiter(s))
                    self.advance()
                case _:
                    raise TypeError()
        return tokens


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

        if start_index.value > len(self.name) - 1:
            raise IndexError()
        elif end_index.value <= start_index.value:
            raise IndexError()
        elif end_index.value > len(self.name):
            raise IndexError()
        else:
            string_slice = self.name[start_index.value : end_index.value]
            return Variable(string_slice)


@dataclass
class Var:
    name: str
    value: str | int | bool = None


@dataclass
class If:
    cond: "AST"
    true_branch: "AST"
    false_branch: "AST"


# Implementing the BoolLiteral Type
@dataclass
class BoolLiteral:
    value: bool

    def __init__(self, val: bool):
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


AST = NumLiteral | BinOp | Variable | Let | BoolLiteral | If | Var

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
        case Var(name, value):
            return (
                "type boolean" * (type(value) == bool)
                + "type string" * (type(value) == str)
                + "type Fraction" * (type(value) == Fraction)
            )
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
        case BinOp("%", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type Fraction"
            else:
                raise TypeError()
        case BinOp("^", left, right):
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
            flag = 1
            for var in variable_list:
                if var.name == left.name:
                    flag = 0
                    if type(var.value) == type(eval(right)):
                        return
            if flag:
                return
            raise TypeError("variables cannot change type")

        case If(cond, true_branch, false_branch):
            if typeof(cond) != "type boolean":
                raise TypeError("Invalid condition")
            return
    raise TypeError()


def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    # typeof(program)
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
        case Var(name, value):
            for variable in variable_list:
                if variable.name == name:
                    return variable.value
            raise InvalidProgram("reference before assignment")
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
        case BinOp("%", left, right):
            return eval(left) % eval(right)
        case If(cond, true_branch, false_branch):
            if eval(cond, environment):
                return eval(true_branch, environment)
            else:
                return eval(false_branch, environment)
        case BinOp("^", left, right):
            return eval(left) ** eval(right)
        case BinOp("!=", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)
        case BinOp("=", left, right):
            right_eval = eval(right)
            match left:
                case Var(name, value):
                    left.value = right_eval
                    flag = 1
                    for var in variable_list:
                        if var.name == left.name:
                            var.value = left.value
                            flag = 0
                    if flag:
                        variable_list.append(left)
                case _:
                    raise InvalidProgram()
        # adding case for print statement
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value
        case _:
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


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token = None
        self.position = -1
        self.advance()

    def advance(self):
        self.position += 1
        self.current_token = (
            self.tokens[self.position] if self.position < len(self.tokens) else None
        )

    def parse_atom(self):
        match self.current_token:
            case Num(value):
                self.advance()
                return NumLiteral(value)
            case Identifier(name):
                self.advance()
                return Var(name=name)
            case Bool(value):
                self.advance()
                return BoolLiteral(value)
            case Paranthesis(paran):
                return self.parse_paran()

    def parse_exp(self):
        left = self.parse_atom()
        while True:
            match self.current_token:
                case Operator(value) if value == "^":
                    self.advance()
                    right = self.parse_atom()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_mul(self):
        left = self.parse_exp()
        while True:
            match self.current_token:
                case Operator(value) if value in "*/%":
                    self.advance()
                    right = self.parse_exp()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_add(self):
        left = self.parse_mul()
        while True:
            match self.current_token:
                case Operator(value) if value in "+-":
                    self.advance()
                    right = self.parse_mul()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_bool(self):
        left = self.parse_add()
        match self.current_token:
            case Operator(value) if value in "> < == != <= >=".split():
                self.advance()
                right = self.parse_add()
                left = BinOp(value, left, right)
        return left

    def parse_assign(self):
        left = self.parse_atom()
        match self.current_token:
            case Operator("="):
                self.advance()
                right = self.parse_bool()
                left = BinOp("=", left, right)
        return left

    def parse_ifelse(self):
        IF = self.parse_bool()
        self.advance()
        COND = self.parse_expr()
        THEN = self.parse_bool()
        self.advance()
        TRUE = self.parse_expr()
        ELSE = self.parse_bool()
        self.advance()
        FALSE = self.parse_expr()
        return If(COND, TRUE, FALSE)

    def parse_expr(self):
        match self.current_token:
            case Keyword(word):
                return self.parse_ifelse()
            case Identifier(name):
                return self.parse_assign()
            case _:
                return self.parse_bool()

    def parse_paran(self):
        assert self.current_token == Paranthesis("(")
        self.advance()
        token_temp = []
        num = 1
        while True:
            s = self.current_token
            if s == Paranthesis("("):
                num += 1
            if s == Paranthesis(")"):
                num = num - 1
            if num == 0:
                break
            token_temp.append(s)
            self.advance()
        self.advance()
        return Parser(token_temp).parse_expr()


def splitter(tokens, masterlist):
    if Delimiter(";") not in tokens:
        if len(tokens) > 0:
            masterlist.append(tokens)
    else:
        masterlist.append(tokens[: tokens.index(Delimiter(";"))])
        splitter(tokens[tokens.index(Delimiter(";")) + 1 :], masterlist)


masterlist = []
l = Lexer("a=4;a+a>9").tokenize()
splitter(l, masterlist)
for token in masterlist:
    print(token)
    print(Parser(token).parse_expr())
    print(eval(Parser(token).parse_expr()))
print(variable_list)

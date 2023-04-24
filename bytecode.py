from dataclasses import dataclass
from fractions import Fraction
from typing import List
from main import AST, BinOp, BoolLiteral, Environment, ForLoop, FunCall, If, Let, Lexer, NumLiteral, Parser, Print, Put, Seq, String, Value, Var, Variable, WhileLoop

@dataclass
class CompiledFunction:
    entry: int

@dataclass
class Label:
    target: int

@dataclass
class UnitType:
    pass

@dataclass
class UnitLiteral:
    pass

class BUG(Exception):
    pass

class RunTimeError(Exception):
    pass

@dataclass
class TypeAssertion:
    expr: "AST"

class I:
    """The instructions for our stack VM."""
    @dataclass
    class PUSH:
        what: Value

    @dataclass
    class UMINUS:
        pass

    @dataclass
    class ADD:
        pass

    @dataclass
    class SUB:
        pass

    @dataclass
    class MUL:
        pass

    @dataclass
    class DIV:
        pass

    @dataclass
    class QUOT:
        pass

    @dataclass
    class REM:
        pass

    @dataclass
    class EXP:
        pass

    @dataclass
    class EQ:
        pass

    @dataclass
    class NEQ:
        pass

    @dataclass
    class LT:
        pass

    @dataclass
    class GT:
        pass

    @dataclass
    class LE:
        pass

    @dataclass
    class GE:
        pass

    @dataclass
    class JMP:
        label: Label

    @dataclass
    class JMP_IF_FALSE:
        label: Label

    @dataclass
    class JMP_IF_TRUE:
        label: Label

    @dataclass
    class NOT:
        pass

    @dataclass
    class DUP:
        pass

    @dataclass
    class PRINT:
        pass

    @dataclass
    class POP:
        pass

    @dataclass
    class LOAD:
        localID: int

    @dataclass
    class STORE:
        localID: int

    @dataclass
    class PUSHFN:
        entry: Label

    @dataclass
    class CALL:
        pass

    @dataclass
    class RETURN:
        pass

    @dataclass
    class HALT:
        pass

Instruction = (
      I.PUSH 
    | I.UMINUS
    | I.ADD
    | I.SUB
    | I.MUL
    | I.DIV
    | I.QUOT
    | I.REM
    | I.JMP
    | I.JMP_IF_FALSE
    | I.JMP_IF_TRUE
    | I.NOT
    | I.DUP
    | I.POP
    | I.LOAD
    | I.STORE
    | I.PUSHFN
    | I.CALL
    | I.RETURN
    | I.HALT
    | I.PRINT
    | I.LT
    | I.GT
    | I.LE
    | I.GE
    | I.EQ
    | I.NEQ
)

@dataclass
class ByteCode:
    insns: List[Instruction]

    def __init__(self):
        self.insns = []

    def label(self):
        return Label(-1)

    def emit(self, instruction):
        self.insns.append(instruction)

    def emit_label(self, label):
        label.target = len(self.insns)
    

class Frame:
    locals: List[Value]
    retaddr: int
    dynamicLink: 'Frame'

    def __init__(self, retaddr = -1, dynamicLink = None):
        MAX_LOCALS = 32
        self.locals = [None] * MAX_LOCALS
        self.retaddr = retaddr
        self.dynamicLink = dynamicLink

class VM:
    bytecode: ByteCode
    ip: int
    data: List[Value]
    currentFrame: Frame

    def load(self, bytecode):
        self.bytecode = bytecode
        self.restart()

    def restart(self):
        self.ip = 0
        self.data = []
        self.currentFrame = Frame()

    def execute(self) -> Value:
        while True:
            assert self.ip < len(self.bytecode.insns)
            match self.bytecode.insns[self.ip]:
                case I.PUSH(val):
                    self.data.append(val)
                    self.ip += 1
                case I.PUSHFN(Label(offset)):
                    self.data.append(CompiledFunction(offset))
                    self.ip += 1
                case I.CALL():
                    self.currentFrame = Frame (
                        retaddr=self.ip + 1,
                        dynamicLink=self.currentFrame
                    )
                    cf = self.data.pop()
                    self.ip = cf.entry
                case I.RETURN():
                    self.ip = self.currentFrame.retaddr
                    self.currentFrame = self.currentFrame.dynamicLink
                case I.UMINUS():
                    op = self.data.pop()
                    self.data.append(-op)
                    self.ip += 1
                case I.ADD():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left+right)
                    self.ip += 1
                case I.SUB():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left-right)
                    self.ip += 1
                case I.MUL():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left*right)
                    self.ip += 1
                case I.DIV():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left/right)
                    self.ip += 1
                case I.EXP():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left**right)
                    self.ip += 1
                case I.QUOT():
                    right = self.data.pop()
                    left = self.data.pop()
                    if left.denominator != 1 or right.denominator != 1:
                        raise RunTimeError()
                    left, right = int(left), int(right)
                    self.data.append(Fraction(left // right, 1))
                    self.ip += 1
                case I.REM():
                    right = self.data.pop()
                    left = self.data.pop()
                    if left.denominator != 1 or right.denominator != 1:
                        raise RunTimeError()
                    left, right = int(left), int(right)
                    self.data.append(Fraction(left % right, 1))
                    self.ip += 1
                case I.EQ():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left==right)
                    self.ip += 1
                case I.NEQ():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left!=right)
                    self.ip += 1
                case I.LT():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left<right)
                    self.ip += 1
                case I.GT():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left>right)
                    self.ip += 1
                case I.LE():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left<=right)
                    self.ip += 1
                case I.GE():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left>=right)
                    self.ip += 1
                case I.JMP(label):
                    self.ip = label.target
                case I.JMP_IF_FALSE(label):
                    op = self.data.pop()
                    if not op:
                        self.ip = label.target
                    else:
                        self.ip += 1
                case I.JMP_IF_TRUE(label):
                    op = self.data.pop()
                    if op:
                        self.ip = label.target
                    else:
                        self.ip += 1
                case I.NOT():
                    op = self.data.pop()
                    self.data.append(not op)
                    self.ip += 1
                case I.DUP():
                    op = self.data.pop()
                    self.data.append(op)
                    self.data.append(op)
                    self.ip += 1
                case I.POP():
                    self.data.pop()
                    self.ip += 1
                case I.LOAD(localID):
                    self.data.append(self.currentFrame.locals[localID])
                    self.ip += 1
                case I.STORE(localID):
                    v = self.data.pop()
                    self.currentFrame.locals[localID] = v
                    self.ip += 1
                case I.PRINT():
                    op = self.data.pop()
                    print(op)
                    self.ip += 1
                case I.HALT():
                    return self.data.pop()


def codegen(program: AST) -> ByteCode:
    code = ByteCode()
    do_codegen(program, code)
    code.emit(I.HALT())
    return code


def do_codegen (
        program: AST,
        code: ByteCode
) -> None:
    def codegen_(program):
        do_codegen(program, code)

    simple_ops = {
        "+": I.ADD(),
        "-": I.SUB(),
        "*": I.MUL(),
        "/": I.DIV(),
        "//": I.QUOT(),
        "%": I.REM(),
        "<": I.LT(),
        ">": I.GT(),
        "<=": I.LE(),
        ">=": I.GE(),
        "==": I.EQ(),
        "!=": I.NEQ(),
        # "not": I.NOT()
    }

    match program:
        
        case NumLiteral(what) | BoolLiteral(what) | Variable(what):
            code.emit(I.PUSH(what))
        case UnitLiteral():
            code.emit(I.PUSH(None))
        case BinOp(op, left, right) if op in simple_ops:
            codegen_(left)
            codegen_(right)
            code.emit(simple_ops[op])
        case BinOp("=", left, right):
            codegen_(right)
            code.emit(I.STORE(left.localID))
        case BinOp("and", left, right):
            E = code.label()
            codegen_(left)
            code.emit(I.DUP())
            code.emit(I.JMP_IF_FALSE(E))
            code.emit(I.POP())
            codegen_(right)
            code.emit_label(E)
        case BinOp("or", left, right):
            E = code.label()
            codegen_(left)
            code.emit(I.DUP())
            code.emit(I.JMP_IF_TRUE(E))
            code.emit(I.POP())
            codegen_(right)
            code.emit_label(E)
        
        case Seq(things):
            if not things: raise BUG()
            last, rest = things[-1], things[:-1]
            for thing in rest:
                codegen_(thing)
                code.emit(I.POP())
            codegen_(last)
        case If(cond, iftrue, iffalse):
            E = code.label()
            F = code.label()
            codegen_(cond)
            code.emit(I.JMP_IF_FALSE(F))
            codegen_(iftrue)
            code.emit(I.JMP(E))
            code.emit_label(F)
            codegen_(iffalse)
            code.emit_label(E)
        case ForLoop(var, start, end, body):
            B = code.label()
            E = code.label()
            codegen_(start)
            code.emit(I.STORE(var.localID))
            code.emit_label(B)
            codegen_(end)
            code.emit(I.LOAD(var.localID))
            code.emit(I.GE())
            code.emit(I.JMP_IF_FALSE(E))
            codegen_(body)
            code.emit(I.LOAD(var.localID))
            code.emit(I.PUSH(1))
            code.emit(I.ADD())
            code.emit(I.STORE(var.localID))
            # code.emit(I.POP())
            code.emit(I.JMP(B))
            code.emit_label(E)
        case WhileLoop(cond, body):
            B = code.label()
            E = code.label()
            code.emit_label(B)
            codegen_(cond)
            # code.emit(I.NEQ())
            code.emit(I.JMP_IF_FALSE(E))
            codegen_(body)
            # code.emit(I.POP())
            code.emit(I.JMP(B))
            code.emit_label(E)
            code.emit(I.PUSH(None))
        case Var() as v:
            code.emit(I.LOAD(v.localID))
        case Put(Var() as v, e):
            codegen_(e)
            code.emit(I.STORE(v.localID))
            code.emit(I.PUSH(None))
        case Let(Var() as v, e1, e2):
            codegen_(e1)
            code.emit(I.STORE(v.localID))
            codegen_(e2)
        case FunCall(fn, _):
            code.emit(I.LOAD(fn.localID))
            code.emit(I.CALL())
        case Print(exp):
            codegen_(exp)
            code.emit(I.DUP())
            code.emit(I.PRINT())
        case TypeAssertion(expr, _):
            codegen_(expr)
        case _:
            if type(program) == list:
                for ast in program:
                    codegen_(ast)


s = input()
text = open(s).read()
l = Lexer(text).tokenize()
pikachu = Parser(l, env = Environment()).mainByte()
# print(pikachu)
CodeList = ByteCode()

for ast in pikachu:
    if type(ast) == list:
        Parser(ast).mainByte()
    else:
        do_codegen(ast, CodeList)

CodeList.emit(I.HALT())
# print(CodeList)
v = VM()
v.load(CodeList)
v.execute()
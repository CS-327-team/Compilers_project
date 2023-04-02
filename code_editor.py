from tkinter import *
from tkinter.filedialog import asksaveasfilename, askopenfilename
import subprocess

compiler = Tk()
compiler.geometry("800x600")


compiler.title('Main IDE')
file_path = ''
current_theme = 'light'

class TextLineNumbers(Canvas):
    def _init_(self, *args, **kwargs):
        Canvas._init_(self, *args, **kwargs)
        self.textwidget = None

    def attach(self, text_widget):
        self.textwidget = text_widget
        
    def redraw(self, *args):
        '''redraw line numbers'''
        self.delete("all")

        i = self.textwidget.index("@0,0")
        while True :
            dline= self.textwidget.dlineinfo(i)
            if dline is None: break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(2,y,anchor="nw", text=linenum)
            i = self.textwidget.index("%s+1line" % i)


class CustomText(Text):
    def _init_(self, *args, **kwargs):
        Text._init_(self, *args, **kwargs, undo=True)

        # create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)

    def _proxy(self, *args):
        # let the actual widget perform the requested action
        cmd = (self._orig,) + args
        result = self.tk.call(cmd)

        # generate an event if something was added or deleted,
        # or the cursor position changed
        if (args[0] in ("insert", "replace", "delete") or 
            args[0:3] == ("mark", "set", "insert") or
            args[0:2] == ("xview", "moveto") or
            args[0:2] == ("xview", "scroll") or
            args[0:2] == ("yview", "moveto") or
            args[0:2] == ("yview", "scroll")
        ):
            self.event_generate("<<Change>>", when="tail")

        # return what the actual widget returned
        return result 

def set_file_path(path):
    global file_path
    file_path = path

def open_file():
    path = askopenfilename(filetypes=[('Custom Files', '*.main')])
    with open(path, 'r') as file:
        code = file.read()
        editor.delete('1.0', END)
        editor.insert('1.0', code)
        set_file_path(path)

def save_as():
    if file_path == '':
        path = asksaveasfilename(filetypes=[('Custom Files', '*.main')])
    else:
        path = file_path
    with open(path, 'w') as file:
        code = editor.get('1.0', END)
        file.write(code)
        set_file_path(path)

def run():
    code_output.delete('1.0', END)
    if file_path == '':
        save_prompt = Toplevel()
        text = Label(save_prompt, text='Please save your code')
        text.pack()
        return
    command = f'python3 main.py {file_path}'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    code_output.insert('1.0', output)
    code_output.insert('1.0',  error)

def change_theme():
    global current_theme
    if current_theme == 'light':
        editor.config(bg='black', fg='white')
        code_output.config(bg='black', fg='white')
        current_theme = 'dark'
    else:
        editor.config(bg='white', fg='black')
        code_output.config(bg='white', fg='black')
        current_theme = 'light'

def save_file(event):
    save_as()


menu_bar = Menu(compiler)

file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Open', command=open_file)
file_menu.add_command(label='Save', command=save_as)
file_menu.add_command(label='Save As', command=save_as)
file_menu.add_command(label='Exit', command=exit)
menu_bar.add_cascade(label='File', menu=file_menu)

view_menu = Menu(menu_bar, tearoff=0)
view_menu.add_command(label='Theme', command=change_theme)
menu_bar.add_cascade(label='View', menu=view_menu)

run_bar = Menu(menu_bar, tearoff=0)
run_bar.add_command(label='Run', command=run)
menu_bar.add_cascade(label='Run', menu=run_bar)

compiler.config(menu=menu_bar)

editor_frame = Frame(compiler)
editor_frame.pack(fill=BOTH, expand=True)

# editor = Text(editor_frame, padx=5, pady=5, wrap='word', undo=True)
editor = CustomText()
vsb = Scrollbar(orient="vertical", command=editor.yview)
editor.configure(yscrollcommand=vsb.set)
editor.tag_configure("bigfont", font=("Helvetica", "24", "bold"))
linenumbers = TextLineNumbers(width=30)
linenumbers.attach(editor)

vsb.pack(side="right", fill="y")
linenumbers.pack(side="left", fill="y")
editor.pack(side="right", fill="both", expand=True)

def _on_change(event):
    linenumbers.redraw()

editor.bind("<<Change>>", _on_change)
editor.bind("<Configure>", _on_change)
editor.bind("<Control-s>", save_file)
editor.bind("<Control-r>", run)
editor.pack(side=TOP, fill=BOTH, expand=True)


code_output = Text(height=10)

editor.pack(fill=BOTH, expand=True)
code_output.pack(fill=BOTH, expand=True)



compiler.mainloop()
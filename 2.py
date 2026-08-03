import tkinter as tk
from tkinter import messagebox


class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Расширенный Калькулятор")
        self.expression = ""

        # Создание интерфейса
        frame = tk.Frame(root)
        frame.pack()

        self.display = tk.Entry(frame, font=('Arial', 20), bd=10, relief=tk.RIDGE, justify='right')
        self.display.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            'C', '0', '=', '+'
        ]

        row_val = 1
        col_val = 0

        for button in buttons:
            action = lambda x=button: self.on_button_click(x)
            tk.Button(frame, text=button, width=5, height=2, command=action).grid(row=row_val, column=col_val, padx=5,
                                                                                  pady=5)
            col_val += 1
            if col_val > 3:
                col_val = 0
                row_val += 1

    def on_button_click(self, char):
        if char == 'C':
            self.expression = ""
            self.display.delete(0, tk.END)
        elif char == '=':
            try:
                result = eval(self.expression)
                self.display.delete(0, tk.END)
                self.display.insert(0, str(result))
                self.expression = str(result)
            except Exception as e:
                messagebox.showerror("Ошибка", "Некорректное выражение")
        else:
            self.expression += char
            self.display.delete(0, tk.END)
            self.display.insert(0, self.expression)


if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()

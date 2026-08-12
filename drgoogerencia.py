"""
LRC Test – однокнопочная проверка теории пространственно-зависимой декогеренции.
С возможностью копирования полного отчёта.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

# Попытка импорта scipy для реальной подгонки
try:
    from scipy.optimize import minimize, curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

C = 299792458.0  # скорость света, м/с

# ============================================================
# ФИЗИЧЕСКАЯ МОДЕЛЬ
# ============================================================
class LRCModel:
    def __init__(self):
        self.tau0 = 1e-6
        self.gamma_env = 0.0
        self.gamma0 = 1e6
        self.time = 1e-6

    @property
    def L0(self):
        return C * self.tau0

    def gamma_lrc(self, dx):
        z = abs(dx) / self.L0
        extra = self.gamma0 * (1.0 - np.exp(-z*z))
        return self.gamma_env + extra

    def gamma_standard(self):
        return self.gamma_env

    def visibility_lrc(self, dx, t=None):
        if t is None:
            t = self.time
        return np.exp(-self.gamma_lrc(dx) * t)

    def visibility_standard(self, t=None):
        if t is None:
            t = self.time
        return np.exp(-self.gamma_standard() * t)

    def generate_data(self, dx_array, noise_std=0.02, true_model='LRC'):
        vis = []
        for dx in dx_array:
            if true_model == 'LRC':
                v = self.visibility_lrc(dx)
            else:
                v = self.visibility_standard()
            v_noisy = v + np.random.normal(0, noise_std)
            v_noisy = np.clip(v_noisy, 0.0, 1.0)
            vis.append(v_noisy)
        return np.array(vis)

# ============================================================
# ФУНКЦИИ ПОДГОНКИ
# ============================================================
def fit_standard(dx, vis, t_obs):
    if not SCIPY_AVAILABLE:
        return 0.1, 1.0, 0, 0
    def residuals(gamma_env):
        return vis - np.exp(-gamma_env * t_obs)
    res = minimize(lambda g: np.sum(residuals(g)**2), x0=[0.1], bounds=[(0, None)])
    gamma_env = res.x[0]
    rss = res.fun
    n = len(dx)
    aic = n * np.log(rss/n) + 2*1
    bic = n * np.log(rss/n) + np.log(n)*1
    return gamma_env, rss, aic, bic

def fit_lrc(dx, vis, t_obs, p0=None):
    if not SCIPY_AVAILABLE:
        return (0.1, 1e6, 1e-6), 1.0, 0, 0
    def model_lrc(dx, gamma_env, gamma0, tau0):
        L0 = C * tau0
        z = abs(dx) / L0
        extra = gamma0 * (1.0 - np.exp(-z*z))
        return np.exp(-(gamma_env + extra) * t_obs)
    try:
        if p0 is None:
            p0 = [0.1, 1e6, 1e-6]
        popt, _ = curve_fit(model_lrc, dx, vis, p0=p0,
                            bounds=([0, 0, 1e-12], [np.inf, np.inf, 10]),
                            maxfev=10000)
        rss = np.sum((vis - model_lrc(dx, *popt))**2)
        n = len(dx)
        aic = n * np.log(rss/n) + 2*len(popt)
        bic = n * np.log(rss/n) + np.log(n)*len(popt)
        return popt, rss, aic, bic
    except Exception:
        return (0.1, 1e6, 1e-6), 1e6, 1e6, 1e6

# ============================================================
# ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ============================================================
class LRCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LRC – однокнопочная проверка + копирование отчёта")
        self.root.geometry("1400x850")

        self.model = LRCModel()
        self.dx_data = None
        self.vis_data = None
        self.fit_results = {}
        self.last_report = ""

        self._build_ui()
        self._update_labels()   # обновляем метки после создания

    def _build_ui(self):
        # ---- Верхняя панель с параметрами ----
        top = ttk.Frame(self.root)
        top.pack(fill='x', padx=10, pady=5)

        # Параметры модели (установлены оптимальные значения)
        ttk.Label(top, text="τ₀ (log10 с)").grid(row=0, column=0, padx=5, sticky='w')
        self.tau_var = tk.DoubleVar(value=-6.0)          # 1e-6 с
        ttk.Scale(top, from_=-15, to=-1, orient='horizontal', variable=self.tau_var, length=150).grid(row=0, column=1, padx=5)
        self.tau_label = ttk.Label(top, text="1e-6")
        self.tau_label.grid(row=0, column=2, padx=5)

        ttk.Label(top, text="Γ_env (log10 1/с)").grid(row=0, column=3, padx=5, sticky='w')
        self.env_var = tk.DoubleVar(value=0.0)           # 1 1/с
        ttk.Scale(top, from_=-2, to=10, orient='horizontal', variable=self.env_var, length=150).grid(row=0, column=4, padx=5)
        self.env_label = ttk.Label(top, text="1.0")
        self.env_label.grid(row=0, column=5, padx=5)

        ttk.Label(top, text="Γ₀ (log10 1/с)").grid(row=0, column=6, padx=5, sticky='w')
        self.g0_var = tk.DoubleVar(value=6.0)            # 1e6 1/с
        ttk.Scale(top, from_=-2, to=12, orient='horizontal', variable=self.g0_var, length=150).grid(row=0, column=7, padx=5)
        self.g0_label = ttk.Label(top, text="1e+6")
        self.g0_label.grid(row=0, column=8, padx=5)

        ttk.Label(top, text="t (log10 с)").grid(row=1, column=0, padx=5, sticky='w')
        self.time_var = tk.DoubleVar(value=-6.0)         # 1e-6 с
        ttk.Scale(top, from_=-12, to=0, orient='horizontal', variable=self.time_var, length=150).grid(row=1, column=1, padx=5)
        self.time_label = ttk.Label(top, text="1e-6")
        self.time_label.grid(row=1, column=2, padx=5)

        # Параметры эксперимента
        ttk.Label(top, text="log10(Δx min)").grid(row=1, column=3, padx=5, sticky='w')
        self.dx_min_var = tk.DoubleVar(value=-2.0)       # 0.01 м
        ttk.Scale(top, from_=-12, to=0, orient='horizontal', variable=self.dx_min_var, length=150).grid(row=1, column=4, padx=5)
        self.dx_min_label = ttk.Label(top, text="0.01")
        self.dx_min_label.grid(row=1, column=5, padx=5)

        ttk.Label(top, text="log10(Δx max)").grid(row=1, column=6, padx=5, sticky='w')
        self.dx_max_var = tk.DoubleVar(value=2.5)        # 316 м
        ttk.Scale(top, from_=-12, to=2, orient='horizontal', variable=self.dx_max_var, length=150).grid(row=1, column=7, padx=5)
        self.dx_max_label = ttk.Label(top, text="316")
        self.dx_max_label.grid(row=1, column=8, padx=5)

        ttk.Label(top, text="Кол-во точек").grid(row=2, column=0, padx=5, sticky='w')
        self.npoints_var = tk.IntVar(value=60)           # 60 точек
        ttk.Scale(top, from_=10, to=100, orient='horizontal', variable=self.npoints_var, length=150).grid(row=2, column=1, padx=5)
        self.npoints_label = ttk.Label(top, text="60")
        self.npoints_label.grid(row=2, column=2, padx=5)

        ttk.Label(top, text="Шум σ").grid(row=2, column=3, padx=5, sticky='w')
        self.noise_var = tk.DoubleVar(value=0.005)       # 0.005
        ttk.Scale(top, from_=0, to=0.1, orient='horizontal', variable=self.noise_var, length=150).grid(row=2, column=4, padx=5)
        self.noise_label = ttk.Label(top, text="0.005")
        self.noise_label.grid(row=2, column=5, padx=5)

        # Истинная модель
        ttk.Label(top, text="Истинная модель:").grid(row=2, column=6, padx=5, sticky='e')
        self.true_model_var = tk.StringVar(value='LRC')
        ttk.Radiobutton(top, text="LRC", variable=self.true_model_var, value='LRC').grid(row=2, column=7, sticky='w')
        ttk.Radiobutton(top, text="Standard", variable=self.true_model_var, value='Standard').grid(row=2, column=8, sticky='w')

        # Привязываем обновление меток к изменению слайдеров
        self.label_pairs = [
            (self.tau_var, self.tau_label),
            (self.env_var, self.env_label),
            (self.g0_var, self.g0_label),
            (self.time_var, self.time_label),
            (self.dx_min_var, self.dx_min_label),
            (self.dx_max_var, self.dx_max_label),
            (self.npoints_var, self.npoints_label),
            (self.noise_var, self.noise_label)
        ]
        for var, label in self.label_pairs:
            var.trace('w', lambda *args, v=var, lbl=label: self._update_single_label(v, lbl))

        # Кнопки
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(btn_frame, text="▶ Запустить проверку", command=self.run_test).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Копировать отчёт", command=self.copy_report).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Сохранить данные", command=self.save_data).pack(side='left', padx=5)

        # ---- Информационное поле ----
        self.info_text = tk.Text(self.root, height=14, font=("Consolas", 9))
        self.info_text.pack(fill='x', padx=10, pady=5)

        # ---- Графики ----
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.ax1.text(0.5, 0.5, "Нажмите «Запустить проверку»", ha='center', va='center', transform=self.ax1.transAxes)
        self.ax2.text(0.5, 0.5, "Результаты появятся здесь", ha='center', va='center', transform=self.ax2.transAxes)
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ОБНОВЛЕНИЯ МЕТОК
    # ------------------------------------------------------------
    def _update_single_label(self, var, label):
        if var == self.npoints_var:
            label.config(text=f"{var.get():.0f}")
        else:
            label.config(text=f"{10**var.get():.2g}")

    def _update_labels(self):
        for var, label in self.label_pairs:
            self._update_single_label(var, label)

    # ------------------------------------------------------------
    # ОСНОВНАЯ ФУНКЦИЯ ПРОВЕРКИ
    # ------------------------------------------------------------
    def run_test(self):
        # Считываем параметры
        self.model.tau0 = 10 ** self.tau_var.get()
        self.model.gamma_env = 10 ** self.env_var.get()
        self.model.gamma0 = 10 ** self.g0_var.get()
        self.model.time = 10 ** self.time_var.get()
        dx_min = 10 ** self.dx_min_var.get()
        dx_max = 10 ** self.dx_max_var.get()
        n_points = int(self.npoints_var.get())
        noise = self.noise_var.get()
        true_model = self.true_model_var.get()

        # Генерация данных
        self.dx_data = np.logspace(np.log10(dx_min), np.log10(dx_max), n_points)
        self.vis_data = self.model.generate_data(self.dx_data, noise, true_model)

        # Подгонка
        t_obs = self.model.time
        if SCIPY_AVAILABLE:
            gamma_env_std, rss_std, aic_std, bic_std = fit_standard(self.dx_data, self.vis_data, t_obs)
            p0 = [self.model.gamma_env, self.model.gamma0, self.model.tau0]
            popt_lrc, rss_lrc, aic_lrc, bic_lrc = fit_lrc(self.dx_data, self.vis_data, t_obs, p0)
        else:
            gamma_env_std, rss_std, aic_std, bic_std = 0.1, 1e6, 1e6, 1e6
            popt_lrc = (0.1, 1e6, 1e-6)
            rss_lrc, aic_lrc, bic_lrc = 1e6, 1e6, 1e6

        self.fit_results = {
            'standard': {'gamma_env': gamma_env_std, 'rss': rss_std, 'aic': aic_std, 'bic': bic_std},
            'lrc': {'gamma_env': popt_lrc[0], 'gamma0': popt_lrc[1], 'tau0': popt_lrc[2],
                    'rss': rss_lrc, 'aic': aic_lrc, 'bic': bic_lrc}
        }

        # ---- ФОРМИРОВАНИЕ ПОЛНОГО ОТЧЁТА ----
        lines = []
        lines.append("="*60)
        lines.append("ОТЧЁТ ПО ПРОВЕРКЕ LRC")
        lines.append("="*60)
        lines.append("")
        lines.append("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА:")
        lines.append(f"  Истинная модель: {true_model}")
        lines.append(f"  Диапазон Δx: от {dx_min:.3e} до {dx_max:.3e} м (log10: {self.dx_min_var.get():.1f} ... {self.dx_max_var.get():.1f})")
        lines.append(f"  Количество точек: {n_points}")
        lines.append(f"  Уровень шума σ: {noise:.3f}")
        lines.append(f"  Время наблюдения t: {t_obs:.3e} с")
        lines.append("")
        lines.append("ПАРАМЕТРЫ МОДЕЛИ (заданные):")
        lines.append(f"  τ₀ = {self.model.tau0:.3e} с")
        lines.append(f"  Γ_env = {self.model.gamma_env:.3e} 1/с")
        lines.append(f"  Γ₀ = {self.model.gamma0:.3e} 1/с")
        lines.append(f"  L₀ = c·τ₀ = {self.model.L0:.3e} м")
        lines.append("")
        lines.append("РЕЗУЛЬТАТЫ ПОДГОНКИ:")
        lines.append("-"*60)
        lines.append("Стандартная модель (γ_env только):")
        lines.append(f"  γ_env = {gamma_env_std:.6e} 1/с")
        lines.append(f"  RSS = {rss_std:.6f}")
        lines.append(f"  AIC = {aic_std:.2f}")
        lines.append(f"  BIC = {bic_std:.2f}")
        lines.append("")
        lines.append("LRC модель (γ_env, γ₀, τ₀):")
        lines.append(f"  γ_env = {popt_lrc[0]:.6e} 1/с")
        lines.append(f"  γ₀ = {popt_lrc[1]:.6e} 1/с")
        lines.append(f"  τ₀ = {popt_lrc[2]:.6e} с")
        lines.append(f"  RSS = {rss_lrc:.6f}")
        lines.append(f"  AIC = {aic_lrc:.2f}")
        lines.append(f"  BIC = {bic_lrc:.2f}")
        lines.append("")
        lines.append("СРАВНЕНИЕ МОДЕЛЕЙ:")
        delta_aic = aic_lrc - aic_std
        delta_bic = bic_lrc - bic_std
        lines.append(f"  ΔAIC = AIC(LRC) – AIC(Standard) = {delta_aic:.2f}")
        lines.append(f"  ΔBIC = BIC(LRC) – BIC(Standard) = {delta_bic:.2f}")
        if delta_aic < -10:
            lines.append("  Интерпретация: ΔAIC < -10 → очень сильное свидетельство в пользу LRC.")
        elif delta_aic < -2:
            lines.append("  Интерпретация: -10 < ΔAIC < -2 → умеренное свидетельство в пользу LRC.")
        elif delta_aic < 2:
            lines.append("  Интерпретация: |ΔAIC| < 2 → модели практически равноценны.")
        else:
            lines.append("  Интерпретация: ΔAIC > 2 → свидетельство в пользу стандартной модели.")
        if not SCIPY_AVAILABLE:
            lines.append("")
            lines.append("⚠️  ВНИМАНИЕ: scipy не установлен, подгонка выполнена упрощённо (результаты демонстрационные).")
        lines.append("="*60)

        report = "\n".join(lines)
        self.last_report = report
        self.info_text.delete('1.0', 'end')
        self.info_text.insert('1.0', report)

        # ---- ПОСТРОЕНИЕ ГРАФИКОВ ----
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.semilogx(self.dx_data, self.vis_data, 'o', color='black', label='Данные')
        v_std = np.exp(-gamma_env_std * t_obs)
        self.ax1.axhline(v_std, linestyle='--', label=f'Standard QM (γ={gamma_env_std:.3g})')
        if SCIPY_AVAILABLE:
            v_lrc = np.array([self._model_lrc(dx, *popt_lrc, t_obs) for dx in self.dx_data])
        else:
            v_lrc = np.exp(-0.1 * t_obs) * np.ones_like(self.dx_data)
        self.ax1.semilogx(self.dx_data, v_lrc, '-', linewidth=2, label='LRC подгонка')
        self.ax1.set_xlabel('Δx (м)')
        self.ax1.set_ylabel('Видимость V')
        self.ax1.grid(alpha=0.25)
        self.ax1.legend(loc='best')
        self.ax1.set_title('Данные и подогнанные модели')

        # Остатки
        if SCIPY_AVAILABLE and 'lrc' in self.fit_results:
            params = self.fit_results['lrc']
            v_lrc_res = np.array([self._model_lrc(dx, params['gamma_env'], params['gamma0'], params['tau0'], t_obs)
                                 for dx in self.dx_data])
            residuals_lrc = self.vis_data - v_lrc_res
        else:
            residuals_lrc = np.zeros_like(self.dx_data)

        if 'standard' in self.fit_results:
            gamma_env = self.fit_results['standard']['gamma_env']
            v_std_res = np.exp(-gamma_env * t_obs)
            residuals_std = self.vis_data - v_std_res
        else:
            residuals_std = np.zeros_like(self.dx_data)

        self.ax2.semilogx(self.dx_data, residuals_std, 's', label='Standard QM', alpha=0.7)
        self.ax2.semilogx(self.dx_data, residuals_lrc, 'o', label='LRC', alpha=0.7)
        self.ax2.axhline(0, linestyle=':', color='gray')
        self.ax2.set_xlabel('Δx (м)')
        self.ax2.set_ylabel('Остатки')
        self.ax2.grid(alpha=0.25)
        self.ax2.legend(loc='best')
        self.ax2.set_title('Остатки моделей')

        self.canvas.draw_idle()

    @staticmethod
    def _model_lrc(dx, gamma_env, gamma0, tau0, t_obs):
        L0 = C * tau0
        z = abs(dx) / L0
        extra = gamma0 * (1.0 - np.exp(-z*z))
        return np.exp(-(gamma_env + extra) * t_obs)

    # ------------------------------------------------------------
    # КОПИРОВАНИЕ ОТЧЁТА
    # ------------------------------------------------------------
    def copy_report(self):
        if not self.last_report:
            messagebox.showinfo("Нет отчёта", "Сначала запустите проверку (кнопка «Запустить проверку»).")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_report)
        self.root.update()  # сохраняет в буфере
        messagebox.showinfo("Скопировано", "Отчёт скопирован в буфер обмена.\nВставьте его в диалог с экспертом.")

    # ------------------------------------------------------------
    # СОХРАНЕНИЕ ДАННЫХ
    # ------------------------------------------------------------
    def save_data(self):
        if self.dx_data is None:
            messagebox.showerror("Ошибка", "Нет данных. Сначала запустите проверку.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file:
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["dx", "visibility"])
                for dx, v in zip(self.dx_data, self.vis_data):
                    writer.writerow([dx, v])
            messagebox.showinfo("Сохранено", f"Данные сохранены в {file}")

# ============================================================
# ЗАПУСК
# ============================================================
def main():
    root = tk.Tk()
    app = LRCApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
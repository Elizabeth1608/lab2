# Задание 1. Интерполяционный многочлен Лагранжа
# Вариант 30: f(x) = cos(sin^2(x))

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# 1. Функция
def f(x):
    return np.cos(np.sin(x) ** 2)

# 2. Параметры
a, b = -np.pi / 2, np.pi / 2
n = 11

print("Интерполяция многочленом Лагранжа")
print(f"Функция:  f(x) = cos(sin^2(x))")
print(f"Отрезок:  [{a:.4f}, {b:.4f}]")
print(f"Степень:  n = {n}")
print(f"Узлов:    {n + 1}")

# 3. Узлы интерполяции
x_nodes = np.linspace(a, b, n + 1)
y_nodes = f(x_nodes)

# 4. Полином Лагранжа
def lagrange_polynomial(x, x_nodes, y_nodes):
    n_pts = len(x_nodes)
    result = np.zeros_like(x, dtype=float)
    for i in range(n_pts):
        li = np.ones_like(x, dtype=float)
        for j in range(n_pts):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return result

x_plot = np.linspace(a, b, 1000)
y_true = f(x_plot)
y_lagrange = lagrange_polynomial(x_plot, x_nodes, y_nodes)

# 5. Практическая погрешность
eps_practical = np.max(np.abs(y_true - y_lagrange))

print(f"\n[1] Практическая погрешность:")
print(f"    eps_практ = sup |f(x) - L_n(x)| = {eps_practical:.6e}")

# 6. Теоретическая погрешность
print(f"\n[2] Теоретическая погрешность:")

# Золотая середина: M = 1e6
M = 1e6
print(f"    M_{n+1} = sup |f^({n+1})(x)| = {M:.1e}")

product = np.ones_like(x_plot)
for xi in x_nodes:
    product *= (x_plot - xi)

R_theor = (M / factorial(n + 1)) * np.abs(product)
eps_theoretical = np.max(R_theor)

print(f"    eps_теор = sup |R_n(x)| = {eps_theoretical:.6e}")

# 7. Графики
plt.figure(figsize=(13, 9))

# Верхний график (интерполяция)
plt.subplot(2, 1, 1)
plt.plot(x_plot, y_true, 'b-', label='f(x) = cos(sin²x)', linewidth=2)
plt.plot(x_plot, y_lagrange, 'r--', label=f'L_{n}(x) — полином Лагранжа',
         linewidth=2, dashes=(6, 3))
plt.plot(x_nodes, y_nodes, 'ko', label='Узлы интерполяции',
         markersize=7, markerfacecolor='yellow')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Интерполяция f(x) = cos(sin²x), n = {n}', fontsize=13)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)

# Нижний график (погрешности)
plt.subplot(2, 1, 2)
plt.plot(x_plot, np.abs(y_true - y_lagrange), 'g-',
         label='Практическая: |f(x) − L_n(x)|', linewidth=2)
plt.plot(x_plot, R_theor, 'm--',
         label='Теоретическая оценка: |R_n(x)|', linewidth=2, dashes=(6, 3))
plt.axhline(eps_practical, color='green', linestyle=':',
            label=f'eps_практ = {eps_practical:.2e}')
plt.axhline(eps_theoretical, color='magenta', linestyle=':',
            label=f'eps_теор = {eps_theoretical:.2e}')
plt.xlabel('x', fontsize=12)
plt.ylabel('Погрешность', fontsize=12)
plt.title('Погрешности интерполяции', fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)

# Обрезаем ось Y, чтобы было видно горбы, но оставляем место для фиолетовой линии
plt.ylim(0, eps_theoretical * 1.2) 

plt.tight_layout()
plt.savefig('lagrange_plot.png', dpi=150, bbox_inches='tight')
print("\n[3] График сохранён: lagrange_plot.png")

# 8. Итоговый отчёт
print("Итоговый отчёт")
print(f"Функция:            f(x) = cos(sin²x)")
print(f"Отрезок:            [{a:.4f}, {b:.4f}]")
print(f"Степень полинома:   n = {n}")
print(f"Число узлов:        {n + 1}")
print(f"Практическая погрешность:   eps_практ = {eps_practical:.6e}")
print(f"Теоретическая погрешность:  eps_теор  = {eps_theoretical:.6e}")
print(f"Отношение eps_теор / eps_практ = {eps_theoretical / eps_practical:.2f}")

plt.show()
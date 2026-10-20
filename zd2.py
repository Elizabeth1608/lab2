# Задание 2. Интерполяция по формуле Гаусса (n = 2)
# Вариант 30: f(x) = cos(sin^2(x))
# Требования: глобальная сетка для поиска узла, локальная сетка (шаг пополам)

import numpy as np

# 1. Функция
def f(x):
    return np.cos(np.sin(x) ** 2)

# 2. Параметры
a, b = -np.pi / 2, np.pi / 2   # отрезок
x0 = 0.3                        # точка, в которой ищем значение
eps = 1e-4                      # заданная точность

print("Интерполяция по формуле Гаусса (n = 2)")
print(f"Функция:        f(x) = cos(sin^2(x))")
print(f"Отрезок:        [{a:.4f}, {b:.4f}]")
print(f"Точка x0:       {x0}")
print(f"Точность eps:   {eps:.1e}")

# 3. Оценка M3 = sup |f'''(x)| (максимум третьей производной)
def numerical_third_derivative(x, h=1e-3):
    return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)

x_check = np.linspace(a + 0.1, b - 0.1, 200)
M3 = np.max(np.abs(numerical_third_derivative(x_check)))
print(f"\n[1] Оценка M3 = sup |f'''(x)| ≈ {M3:.4f}")

# 4. Подбор шага глобальной сетки H
# Остаточный член для Гаусса: |R_2| = M3 * h^3 * |t(t^2-1)| / 6
# Максимум |t(t^2-1)|/6 на отрезке [-0.5, 0.5] достигается при t = ±1/sqrt(3) и равен ~0.0481
H_max = (eps / (M3 * 0.0481)) ** (1/3)
H = H_max * 0.5  # берем шаг с запасом (в 2 раза меньше максимального)
print(f"[2] Максимальный шаг по оценке: H_max = {H_max:.6f}")
print(f"    Выбранный шаг глобальной сетки: H = {H:.6f}")

# 5. Строим ГЛОБАЛЬНУЮ сетку и ищем ближайший узел к x0
x_global = np.arange(a, b + H, H)
j = np.argmin(np.abs(x_global - x0))

# Защита от выхода за границы массива
if j == 0: j = 1
if j >= len(x_global) - 1: j = len(x_global) - 2

x_j_global = x_global[j]
print(f"[3] Ближайший узел глобальной сетки: x_j = {x_j_global:.6f}")

# 6. Строим ЛОКАЛЬНУЮ сетку (шаг пополам) вокруг x_j
h_local = H / 2
print(f"[4] Шаг локальной сетки: h = H/2 = {h_local:.6f}")

# Берем три точки: слева, по центру (середина интервала) и справа
x_mid = x_j_global
x_left = x_j_global - h_local
x_right = x_j_global + h_local

y_mid = f(x_mid)
y_left = f(x_left)
y_right = f(x_right)

print(f"[5] Узлы для формулы Гаусса (локальная сетка):")
print(f"    x_(j-1) = {x_left:.6f}, y_(j-1) = {y_left:.6f}")
print(f"    x_j     = {x_mid:.6f}, y_j     = {y_mid:.6f} (середина)")
print(f"    x_(j+1) = {x_right:.6f}, y_(j+1) = {y_right:.6f}")

# 7. Вычисляем конечные разности
y_first  = y_right - y_mid
y_second = y_right - 2*y_mid + y_left

print(f"[6] Конечные разности:")
print(f"    Первая:  y_(j+1/2)^I = {y_first:.6f}")
print(f"    Вторая:  y_j^2       = {y_second:.6f}")

# 8. Параметр t (считаем относительно локального шага h_local)
t = (x0 - x_mid) / h_local
print(f"[7] Параметр t = (x0 - x_j) / h = {t:.6f}")

# 9. Вычисление по формуле Гаусса
L2 = y_mid + y_first * t + y_second * t * (t - 1) / 2
print(f"[8] Значение по формуле Гаусса: L_2(x0) = {L2:.8f}")

# 10. Сравнение с точным значением и проверка точности
y_exact = f(x0)
error = abs(y_exact - L2)

print(f"\n[9] Точное значение: f(x0) = {y_exact:.8f}")
print(f"    Практическая погрешность: |f(x0) - L_2(x0)| = {error:.2e}")

if error <= eps:
    print(f"    -> Точность ДОСТИГНУТА ({error:.2e} <= {eps:.2e})")
else:
    print(f"    -> Точность НЕ ДОСТИГНУТА ({error:.2e} > {eps:.2e})")

# 11. Теоретическая оценка остаточного члена для x0
R2 = M3 * h_local**3 * abs(t * (t**2 - 1)) / 6
print(f"[10] Теоретическая оценка остаточного члена: |R_2(x0)| = {R2:.6e}")

print("\nИтог:")
print(f"f({x0}) ≈ {L2:.8f} (точное: {y_exact:.8f}, погрешность: {error:.2e})")
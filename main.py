import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def linear_approximation(x, y):
    n = len(x)
    SX = sum(x)
    SXX = sum(i ** 2 for i in x)
    SY = sum(y)
    SXY = sum(x[i] * y[i] for i in range(n))

    delta = n * SXX - SX ** 2
    a = (n * SXY - SX * SY) / delta
    b = (SXX * SY - SX * SXY) / delta

    func = lambda x: a * x + b
    return func, a, b


def quadratic_approximation(x, y):
    n = len(x)
    SX = sum(x)
    SX2 = sum(p ** 2 for p in x)
    SX3 = sum(p ** 3 for p in x)
    SX4 = sum(p ** 4 for p in x)
    SY = sum(y)
    SXY = sum(x[i] * y[i] for i in range(n))
    SX2Y = sum(x[i] ** 2 * y[i] for i in range(n))

    x_matrix = np.array([[n, SX, SX2], [SX, SX2, SX3], [SX2, SX3, SX4]])
    y_vector = np.array([SY, SXY, SX2Y])
    coefficients = np.linalg.solve(x_matrix, y_vector)

    func = lambda x: coefficients[2] * x ** 2 + coefficients[1] * x + coefficients[0]

    return func, coefficients[2], coefficients[1], coefficients[0]


def cubic_approximation(x, y):
    n = len(x)
    SX = sum(x)
    SX2 = sum(p ** 2 for p in x)
    SX3 = sum(p ** 3 for p in x)
    SX4 = sum(p ** 4 for p in x)
    SX5 = sum(p ** 5 for p in x)
    SX6 = sum(p ** 6 for p in x)
    SY = sum(y)
    SXY = sum(x[i] * y[i] for i in range(n))
    SX2Y = sum(x[i] ** 2 * y[i] for i in range(n))
    SX3Y = sum(x[i] ** 3 * y[i] for i in range(n))

    x_matrix = np.array([[n, SX, SX2, SX3], [SX, SX2, SX3, SX4], [SX2, SX3, SX4, SX5], [SX3, SX4, SX5, SX6]])
    y_vector = np.array([SY, SXY, SX2Y, SX3Y])
    coefficients = np.linalg.solve(x_matrix, y_vector)

    func = lambda x: coefficients[3] * x ** 3 + coefficients[2] * x ** 2 + coefficients[1] * x + coefficients[0]

    return func, coefficients[3], coefficients[2], coefficients[1], coefficients[0]


def exponential_approximation(x, y):
    if not all([i > 0 for i in x]):
        return
    y_ln = [np.log(i) for i in y]
    function, B, A = linear_approximation(x, y_ln)
    a = np.exp(A)
    b = B
    func = lambda x: a * np.exp(b * x)
    return func, a, b


def logarithmic_approximation(x, y):
    if not all([i > 0 for i in x]):
        return
    x_ln = [np.log(i) for i in x]
    function, a, b = linear_approximation(x_ln, y)
    func = lambda x: a * np.log(x) + b
    return func, a, b


def power_approximation(x, y):
    if not (all([p > 0 for p in x]) and all([p > 0 for p in y])):
        return
    x_ln = [np.log(p) for p in x]
    y_ln = [np.log(p) for p in y]
    function, B, A = linear_approximation(x_ln, y_ln)
    a = np.exp(A)
    b = B
    func = lambda x: a * x ** b
    return func, a, b


def read_data_from_file(filename):
    points = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                points.append((x, y))
        num_points = len(points)
        if num_points <= 8 or num_points >= 12:
            print("Ошибка: количество точек в файле должно быть от 8 до 12.")
            return []
        return points
    except FileNotFoundError:
        print("Файл с данными не найден.")
        return []


def get_S(x, y, func):
    return sum([(func(x[i]) - y[i]) ** 2 for i in range(len(x))])


def mean_squared_error(x, y, func):
    return np.sqrt(get_S(x, y, func) / len(x))


def calculate_pearson_correlation(x_values, y_values):
    mean_x = sum([p for p in x_values]) / len(x_values)
    mean_y = sum([p for p in y_values]) / len(y_values)
    return (sum([(x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(len(x_values))]) /
            np.sqrt(sum([(p - mean_x) ** 2 for p in x_values]) * sum([(p - mean_y) ** 2 for p in y_values])))


def coefficient_of_determination(x, y, func):
    phi = sum([func(i) for i in x]) / len(x)
    return 1 - (sum([(y[i] - func(x[i])) ** 2 for i in range(len(x))]) / sum([(p - phi) ** 2 for p in y]))


def write_results_to_file(file_name, approximation, coefficients, mse, r_squared):
    with open(file_name, 'a') as file:
        file.write(f'Наилучшая аппроксимирующая функция: {approximation}\n')
        file.write(f'Коэффициенты аппроксимирующей функции: {coefficients}\n')
        file.write(f'Среднеквадратичное отклонение: {mse}\n')
        file.write(f'Коэффициент детерминации: {r_squared}\n')


def clear_file(file_name):
    with open(file_name, 'w') as file:
        pass

def input_data(points_counter):
    points = []
    while len(points) < points_counter:
        try:
            x = float(input("Введите значение x: "))
            y = float(input("Введите значение y: "))
            if any(point[0] == x and point[1] == y for point in points):
                print("Точка уже была введена. Пожалуйста, введите уникальные значения.")
                continue
            else:
                print(f"Запомнил точку ({x} {y})")
                points.append((x, y))
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите числовое значение.")
    return points


def main():
    print("Выберите источник данных:")
    print("1. Из файла")
    print("2. Ввод с клавиатуры")
    choice = input("Ваш выбор: ")

    if choice == "1":
        filename = input("Введите имя файла с данными: ")
        points = read_data_from_file(filename)
        if points:
            print("Данные из файла:", points)
        else:
            print("Произошла ошибка при чтении из файла.")
    elif choice == "2":
        points_counter = int(input("Введите количество точек (от 8 до 12): "))
        if points_counter <= 8 or points_counter >= 12:
            print("Некорректное количество точек. Пожалуйста, введите число от 8 до 12.")
            return
        print("Введите координаты точек (x, y).")
        points = input_data(points_counter)
        print("Введенные точки:", points)
    else:
        print("Некорректный выбор.")

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    f1, a1, b1 = linear_approximation(x, y)
    f2, a2, b2, c2 = quadratic_approximation(x, y)
    f3, a3, b3, c3, d3 = cubic_approximation(x, y)
    f4, a4, b4 = exponential_approximation(x, y)
    f5, a5, b5 = logarithmic_approximation(x, y)
    f6, a6, b6 = power_approximation(x, y)

    f = [f1, f2, f3, f4, f5, f6]
    v = ["ax + b", "ax^2 + bx + c", "ax^3 + bx^2 + cx + d", "ae^(bx)", "alnx + b", "ax^b"]
    a = [a1, a2, a3, a4, a5, a6]
    b = [b1, b2, b3, b4, b5, b6]
    c = ["-", c2, c3, "-", "-", "-"]
    d = ["-", "-", d3, "-", "-", "-"]

    titles = ["Линейная аппроксимация", "Квадратичная аппроксимация", "Кубическая аппроксимация",
              "Экспоненциальная аппроксимация", "Логарифмическая аппроксимация", "Степенная аппроксимация"]

    res = PrettyTable()
    res.title = "Выбор аппроксимирующей функции"
    res.field_names = ["Вид функции", "a", "b", "c", "d", "S", "stand dev", "R^2"]
    res.float_format = ".5"

    for i in range(6):
        table = PrettyTable()
        table.title = titles[i]
        table.field_names = ["№", "X", "Y", "Phi", "eps"]
        table.float_format = ".3"
        if f[i] is None:
            continue
        for j in range(len(x)):
            table.add_row([j + 1, x[j], y[j], f[i](x[j]), f[i](x[j]) - y[j]])
        print("\n", table)
        r2 = coefficient_of_determination(x, y, f[i])
        res.add_row([v[i], a[i], b[i], c[i], d[i], get_S(x, y, f[i]), mean_squared_error(x, y, f[i]), r2])
        print(f"Коэффициент детерминации: {r2:.5f}")
        if r2 < 0.5:
            print("Точность аппроксимации недостаточна")
        elif r2 < 0.75:
            print("Слабая аппроксимация")
        elif r2 < 0.95:
            print("Удовлетворительная аппроксимация")
        else:
            print("Высокая точность аппроксимации")

        if i == 0:
            pr = calculate_pearson_correlation(x, y)
            print(f"Коэффициент Пирсона: {pr:.5f}")
            if pr == 0:
                print("Связь между переменными отсутствует")
            elif pr == 1 or pr == -1:
                print("Строгая линейная зависимость")
            elif pr < 0.3:
                print("Связь слабая")
            elif pr < 0.5:
                print("Связь умеренная")
            elif pr < 0.7:
                print("Связь заметная")
            elif pr < 0.9:
                print("Связь высокая")
            else:
                print("Связь весьма высокая")

    print("\n", res)

    mses = []
    approximations = ["Линейная", "Квадратичная", "Кубическая", "Экспоненциальная", "Логарифмическая", "Степенная"]
    functions = [f1, f2, f3, f4, f5, f6]

    for f in functions:
        if f is not None:
            mse = mean_squared_error(x, y, f)
            mses.append(mse)
        else:
            mses.append(np.inf)

    epsilon = 1e-9
    output_file_name = input("Введите название файла для записи ответа: ")
    clear_file(output_file_name)

    min_mse = np.min(mses)

    best_approximation_indices = np.where(np.abs(mses - min_mse) < epsilon)[0]

    for best_approximation_index in best_approximation_indices:
        best_approximation = approximations[best_approximation_index]
        print(f"Наилучшая аппроксимирующая функция: {best_approximation}")

        coefficients = [a[best_approximation_index], b[best_approximation_index], c[best_approximation_index],
                        d[best_approximation_index]]
        mse = mses[best_approximation_index]
        r_squared = coefficient_of_determination(x, y, functions[best_approximation_index])

        # print(f"Коэффициенты: {coefficients}")
        # print(f"MSE: {mse}")
        # print(f"R^2: {r_squared}")
        write_results_to_file(output_file_name, best_approximation, coefficients, mse, r_squared)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Исходные данные')

    for i, f in enumerate(functions):
        if f is not None:
            if i in [3, 4, 5]:
                x_ = np.linspace(min(x), max(x) + 1, 1000)
            else:
                x_ = np.linspace(min(x) - 1, max(x) + 1, 1000)
            yi = np.array([f(a) for a in x_])
            plt.plot(x_, yi, label=titles[i])

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

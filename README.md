# MACWING - MAC Visualizer / Визуализатор САХ

[RU] Профессиональный инструмент для авиационных инженеров и авиамоделистов, предназначенный для расчета и визуализации Средней Аэродинамической Хорды (САХ) крыльев сложной формы.

[EN] A professional tool for aeronautical engineers and RC hobbyists designed to calculate and visualize the Mean Aerodynamic Chord (MAC) of complex wing shapes.

---

## Features / Возможности

*   **Multi-segment wings / Многосегментные крылья**: Support for complex wings composed of multiple trapezoidal segments. / Поддержка крыльев из нескольких трапециевидных сегментов.
*   **Canard Scheme / Схема «Утка»**: Module for equivalent chord and static margin calculation for canard layouts. / Расчет эквивалентной хорды и центровки для схемы «утка».
*   **Interactive Visualization / Интерактивность**: Real-time plot updates using Matplotlib. / Мгновенное обновление графиков.
*   **Multi-language Support / Мультиязычность**: Built-in support for English and Russian. / Поддержка русского и английского языков.
*   **High Precision / Высокая точность**: Uses `decimal` library for engineering calculations. / Использование `decimal` для точности вычислений.

## Installation / Установка

1.  Clone the repository / Клонируйте репозиторий:
    ```bash
    git clone https://github.com/Nofflll/MAC_WING.git
    cd MAC_WING
    ```

2.  Install dependencies / Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

## Usage / Запуск

Run the main script / Запустите основной файл:
```bash
python MACWING.py
```

## How to calculate / Как пользоваться

1.  **Main Window / Главное окно**: Add wings and segments. Enter chords, span, and sweep. / Добавляйте крылья и сегменты. Вводите хорды, размах и стреловидность.
2.  **Language / Язык**: Click the "Language" button to toggle between RU and EN. / Нажмите кнопку "Language" для переключения между RU и EN.
3.  **Canard / Утка**: Use the "Open Canard Window" for advanced aerodynamic center calculations. / Используйте окно «Утка» для расчета аэродинамического фокуса.

## Math / Математика

The program implements:
- MAC finding via diagonal intersection method. / Нахождение САХ методом диагоналей.
- Recursive segment merging. / Рекурсивное объединение сегментов.
- Equivalent chord for interference effects. / Расчет эквивалентной хорды.

## Author / Автор

[Nofflll]

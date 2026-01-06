import sys
from PyQt5 import QtWidgets
from macwing_ui import WingSACVisualizer

def main():
    """
    Точка входа в приложение MACWING.
    """
    app = QtWidgets.QApplication(sys.argv)
    
    # Настройка стиля (опционально)
    app.setStyle("Fusion")
    
    # Создание и отображение основного окна
    main_win = WingSACVisualizer()
    main_win.show()
    
    # Запуск цикла обработки событий
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

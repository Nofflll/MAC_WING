class Translations:
    RU = {
        "app_title": "MACWING - Визуализатор САХ",
        "add_wing": "Добавить новое крыло",
        "angle": "Угол (°):",
        "pivot_x": "Pivot X:",
        "pivot_y": "Pivot Y:",
        "show_diagonals": "Показывать диагонали",
        "mac_info": "Информация по САХ:",
        "table_headers": ["Видим", "X нач.", "Y нач.", "Корн. хорда", "Конц. хорда", "Стрел.(°)", "Размах"],
        "wing_tab": "Крыло",
        "add_wing_menu": "Добавить новое крыло",
        "remove_wing_menu": "Удалить текущее крыло",
        "add_segment_menu": "Добавить сегмент",
        "remove_segment_menu": "Удалить сегмент",
        "plot_wing_label": "Крыло {w}, этап {s}",
        "sac_info_format": "Крыло {w}, этап {s}, сегм.{seg}: L={L:.6f}, центр=({x:.3f}, {y:.3f})",
        "lang_toggle": "Language: RU"
    }

    EN = {
        "app_title": "MACWING - MAC Visualizer",
        "add_wing": "Add New Wing",
        "angle": "Angle (°):",
        "pivot_x": "Pivot X:",
        "pivot_y": "Pivot Y:",
        "show_diagonals": "Show Diagonals",
        "mac_info": "MAC Information:",
        "table_headers": ["Visible", "X Start", "Y Start", "Root Chord", "Tip Chord", "Sweep(°)", "Span"],
        "wing_tab": "Wing",
        "add_wing_menu": "Add New Wing",
        "remove_wing_menu": "Remove Current Wing",
        "add_segment_menu": "Add Segment",
        "remove_segment_menu": "Remove Segment",
        "plot_wing_label": "Wing {w}, Stage {s}",
        "sac_info_format": "Wing {w}, Stage {s}, Seg.{seg}: L={L:.6f}, center=({x:.3f}, {y:.3f})",
        "lang_toggle": "Language: EN"
    }

class I18n:
    _current_lang = "RU"
    _data = {"RU": Translations.RU, "EN": Translations.EN}

    @classmethod
    def set_lang(cls, lang):
        if lang in cls._data:
            cls._current_lang = lang

    @classmethod
    def get_lang(cls):
        return cls._current_lang

    @classmethod
    def t(cls, key):
        return cls._data[cls._current_lang].get(key, key)

    @classmethod
    def toggle_lang(cls):
        cls._current_lang = "EN" if cls._current_lang == "RU" else "RU"
        return cls._current_lang

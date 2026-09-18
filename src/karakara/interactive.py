"""交互模式用的文件对话框。

``tkinter`` 是**标准库**，所以这不会给主环境增加依赖；但它只在交互模式里被
用到，因此 import 放在函数内部——批处理（CI、真实曲库）不该为了两个对话框
去初始化 Tk。
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Literal


def ask_for_input_file(type_: Literal["lyrics", "audio"]) -> str:
    """弹出「打开文件」对话框，返回所选路径（取消时返回空串）。"""
    import tkinter as tk
    from tkinter import filedialog as fd

    root = tk.Tk()
    root.withdraw()
    if type_ == "lyrics":
        filetypes = [("LRC files", "*.lrc"), ("All files", "*.*")]
    else:
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
            ("All files", "*.*"),
        ]
    file_path = fd.askopenfilename(
        parent=root,
        title=f"Select an {type_} file",
        filetypes=filetypes,
    )
    root.destroy()
    return file_path


def ask_for_output_path(default_file: PathLike[str]) -> str:
    """弹出「另存为」对话框，返回所选路径（取消时返回空串）。"""
    import tkinter as tk
    from tkinter import filedialog as fd

    default_file = Path(default_file).resolve()
    root = tk.Tk()
    root.withdraw()
    file_path = fd.asksaveasfilename(
        parent=root,
        title="Select output file path",
        defaultextension=".lrc",
        filetypes=[("LRC files", "*.lrc"), ("All files", "*.*")],
        initialfile=default_file.name,
        initialdir=default_file.parent,
    )
    root.destroy()
    return file_path

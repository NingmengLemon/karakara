from os import PathLike
from pathlib import Path
from typing import Literal

from karakara.aligner import Qwen3ForcedAligner
from karakara.core import gen_kara
from karakara.logging import setup_logging
from karakara.preprocess import AudioPreprocessConfig
from karakara.separator.demucs import DemucsSeparator
from karakara.spl_parser import Lyrics


def ask_for_input_file(type_: Literal["lyrics", "audio"]) -> str:
    # use tkiner dialog to ask for file path
    import tkinter as tk
    from tkinter import filedialog as fd

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if type_ == "lyrics":
        filetypes = [("LRC files", "*.lrc"), ("All files", "*.*")]
    else:
        filetypes = [("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
    file_path = fd.askopenfilename(
        parent=root,
        title=f"Select an {type_} file",
        filetypes=filetypes,
    )
    root.destroy()
    return file_path


def ask_for_output_path(default_file: PathLike) -> str:
    # use tkiner dialog to ask for output file path
    import tkinter as tk
    from tkinter import filedialog as fd

    default_file = Path(default_file).resolve()
    root = tk.Tk()
    root.withdraw()  # Hide the main window
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


def main() -> None:
    setup_logging()

    lyrics_src = input("lyrics:").strip() or ask_for_input_file("lyrics")
    print("lrc src:", lyrics_src)
    with open(lyrics_src, "r") as fp:
        lyrics = Lyrics.loads(fp.read())

    audio_src = input("audio file:").strip() or ask_for_input_file("audio")
    print("audio src:", audio_src)

    preprocess_cfg = AudioPreprocessConfig(
        normalize=True,
        suppress_vibrato=True,
        compress=False,
    )

    # 传 dump_dir 以保存各阶段中间音频，方便调试；置 None 则不保存
    dump_dir = input("dump dir (blank=skip):").strip() or None

    lyrics = gen_kara(
        lyrics,
        audio_src,
        aligner=Qwen3ForcedAligner(),
        separator=DemucsSeparator(),
        preprocess_config=preprocess_cfg,
        dump_dir=dump_dir,
    )

    save_as = input("output:") or ask_for_output_path(
        Path(lyrics_src).with_suffix(".kara.lrc")
    )
    print("saving as:", save_as)
    with open(save_as, "w+") as fp:
        fp.write(
            lyrics.dumps(
                use_bracket_for_byword_tag=True,
                # compatible with foobar2000
            )
        )


if __name__ == "__main__":
    main()

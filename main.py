from karakara.aligner import GentleAligner
from karakara.core import gen_kara
from karakara.logging import setup_logging
from karakara.preprocess import AudioPreprocessConfig
from karakara.spl_parser import Lyrics

if __name__ == "__main__":
    setup_logging()

    lyrics_src = input("lyrics:").strip() or "samples/spark_for_dream.lrc"
    print("lrc src:", lyrics_src)
    with open(lyrics_src, "r") as fp:
        lyrics = Lyrics.loads(fp.read())

    audio_src = input("audio file:").strip() or "samples/spark_for_dream.mp3"
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
        aligner=GentleAligner(),
        preprocess_config=preprocess_cfg,
        dump_dir=dump_dir,
    )

    save_as = input("output:") or "tmp/spark_for_dream.lrc"
    print("save as:", save_as)
    with open(save_as, "w+") as fp:
        fp.write(lyrics.dumps())

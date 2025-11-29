import sys

if "src" not in sys.path:
    sys.path.insert(0, "src")

from karakara.kara_gen import gen_kara
from karakara.spl_parser import Lyrics
from karakara.utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    lyrics_src = input("lyrics:").strip() or "samples/spark_for_dream.lrc"
    print("lrc src:", lyrics_src)
    with open(lyrics_src, "r") as fp:
        lyrics = Lyrics.loads(fp.read())
    audio_src = input("audio file:").strip() or "samples/spark_for_dream.mp3"
    print("audio src:", audio_src)
    lyrics = gen_kara(lyrics, audio_src)
    save_as = input("output:") or "tmp/spark_for_dream.lrc"
    print("save as:", save_as)
    with open(save_as, "w+") as fp:
        fp.write(lyrics.dumps())

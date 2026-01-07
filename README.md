# karakara

> **Kara**oke & [**KARA**KARA](https://moegirl.icu/KARAKARA)

只是 1 个神秘的 Playground (

于是有参考资料:

- [FunASR](https://github.com/modelscope/FunASR)
- [SPL Format](https://moriafly.com/standards/spl.html)
- [UVR Models](https://github.com/TRvlvr/model_repo/releases/)
- [Demucs](https://github.com/adefossez/demucs)
- [whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/AIXerum/faster-whisper)
- [mfa](https://mfa-models.readthedocs.io/en/latest/index.html)
- [gentle on dockerhub](https://hub.docker.com/r/lowerquality/gentle)

> 已知一首歌的音频文件和行级别精度的歌词文件, 能否用程序做出词级别精度的歌词文件呢...?

唯一花了心思的部分是一个简陋的lrc歌词解析器: [spl_parser](src/karakara/spl_parser.py), 参考 [SPL 歌词格式](https://moriafly.com/standards/spl.html) (2025-11-14 版) ~~, 但其实并没有严格遵循~~

位于 [samples/](samples/) 下的样本文件们的版权归其各自的原始创作者们所有

~~因为只是一个 Playground, 当然是所有可能用到的依赖都装上啦~~

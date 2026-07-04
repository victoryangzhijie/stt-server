# Test fixtures

## `speech_16k_mono_s16le.pcm`

One second of real American English speech, raw PCM16 (signed 16-bit
little-endian), mono, 16 kHz, headerless (32,000 bytes). Used by the Silero
VAD model tests (`tests/core/test_vad_silero.py`) as a speech-positive input,
because Silero — a learned speech detector — does not classify synthetic sine
tones or white noise as speech.

**Source:** Open Speech Repository (Telchemy),
<https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav>
(American English, harvard sentences, 8 kHz WAV). Retrieved 2026-07-03.

**Conversion:** the source 8 kHz mono PCM16 WAV was resampled to 16 kHz
(Python stdlib `audioop.ratecv`), and a single 1-second high-energy segment
(starting at t = 1.0 s of the resampled audio) was extracted and saved as raw
headerless PCM16LE.

**License / usage terms**, as stated on the source site
(<https://www.voiptroubleshooter.com/open_speech/index.html>, retrieved
2026-07-03):

> "The Open Speech Repository provides freely usable speech files in multiple
> languages for use in Voice over IP testing and other applications. [...] We
> encourage you to use these files, publish, copy, broadcast without
> restriction - we only require that you identify the source of files used as
> 'Open Speech Repository'."

and, under "Conditions of use":

> "The material on this site is freely available for use in VoIP testing,
> research, development, marketing and any other reasonable application. The
> material may be copied, downloaded, broadcast, modified, incorporated into
> web sites or test equipment. We do require that you identify the source of
> the speech materials as 'Open Speech Repository'."

**Attribution:** the speech material in `speech_16k_mono_s16le.pcm` is from
the **Open Speech Repository**.

Note: these terms apply to this fixture file only; they are separate from
(and compatible with redistribution alongside) this repository's
Apache-2.0 license, which covers the project's code.

# Qualification speech

`question.wav` is synthesized with Windows Microsoft Zira Desktop (System.Speech), with the text:

> Hello. Please tell me what two plus three equals, and then introduce yourself briefly.

The qualification client resamples it to mono 16 kHz PCM16 before sending one-second chunks to native duplex. Acceptance requires generated speech audio and a correct answer to the spoken question.

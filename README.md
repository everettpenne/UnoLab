# Uno Lab

An experimental companion to Uno Launcher for playing with on-device AI on a Pixel. It is a separate app (`com.everettpenne.unolab`), so a 1.3 GB
model never shares memory with the launcher, and it has no network permission: nothing said to the model leaves the phone.

## What it does

- **Chat** with Qwen3.8-2B (Q4_K_M GGUF) on the CPU through [llama.cpp](https://github.com/ggml-org/llama.cpp) (a submodule in
  `third_party/llama.cpp`, MIT), with tokens streamed as they come, an optional **Think** mode, and after each reply its prefill and decode
  speed and, on battery, the energy per token.
- **NPU lab**: a raw-NNAPI probe (no Tensor SDK, no TFLite) inside the app: the NNAPI devices it can see, a sweep of
  int8 fully-connected layers pinned to `google-edgetpu` with the achieved GB/s and a CPU correctness check, a prefill-shaped (batch 128)
  run in GMAC/s, and a 10-second sustained stream with the draw sampled, for joules per GB.
- **Ask Qwen** from anywhere: select text in any app and pick *Ask Qwen*, share text to it, or send
  `com.everettpenne.unolab.ASK` with the question in `Intent.EXTRA_TEXT` (for Uno's search to hand a query over). A question that arrives
  while the model is loading is asked once it is ready.

Why the chat is on the CPU and not the NPU: measured on a Pixel 10a, the NPU streams weights at 20–26 GB/s for about a third of the
CPU's energy per token, but it takes int8 only, and int8 activations wreck Qwen3.5's Gated DeltaNet layers by the third layer.

## The assistant service (for Uno Launcher)

`AssistantService` is a bound service (`IAssistant.aidl`) that Uno's search can ask, so the model stays in this app's process and never in
the launcher's. It serves only Uno and Uno Dev, checked on every call by package and signing certificate (`CallerPolicy`). Two modes:
**ANSWER**, a short streamed reply, and **ACTION**, where a GBNF grammar holds the model to one JSON object naming one of Uno's actions
(timer, alarm, stopwatch, torch, focus, note, open an app, call, text, web search, or none). Uno shows an action as a row to tap; nothing is
done on the model's word alone. Each mode's system prompt is decoded once and its state saved, so later questions start from it. The chat
and the service share one copy of the weights (two contexts), freed three minutes after nothing holds it.

**Phone context.** `askWithContext(query, context, mode, sink)` takes, beside the question, a few lines Uno gathers on the phone for that
question (date and time, battery, next alarm, what's playing, and so on), so Qwen can answer "what day is it?" from the phone rather than
guess. The context is cleaned like the query (chat-format tokens and control characters taken out), cut at 2,000 characters, and placed at
the start of the user turn as `<context>\n\nQuestion: <query>`, after the saved system prompt, so that prompt's state is still reused. Both
prompts tell the model to use it when it answers the question and never to invent it. It is used for that one question and kept nowhere;
Uno Lab still has no network permission. `version()` answers 1; an Uno Lab from before this has no `version()` (the call returns 0), and Uno
then falls back to `ask()` without context.

On the one-core emulator, "pasta timer 12 minutes" came back as `{"action":"timer","seconds":720,"label":"pasta"}` in 52 s (most of it the
system prompt), and the next question, with the prompt's saved state, in 7.5 s. Not yet timed on a phone.

## Install

Download `UnoLab-<version>.apk` from the [Releases](https://github.com/everettpenne/UnoLab/releases) page and check it against
`SHA256SUMS.txt`. It is arm64 only, needs Android 12 or later, and is meant for Pixels with a Tensor chip (the NPU lab needs the
`google-edgetpu` NNAPI driver; the chat runs on any arm64 phone with about 4 GB free). The APK has no model in it.

## Getting the model on

Push it to the app's drop folder (no permission needed), then open the app; it moves the file into its internal storage and loads it from there:

```sh
adb push Qwen3.8-2B-Q4_K_M.gguf /sdcard/Android/data/com.everettpenne.unolab/files/
```

Or use **Import model** on the Model tab to pick a `.gguf` with the system picker. One model is kept at a time.

The drop folder is on shared storage, which Android serves through FUSE: memory-mapped from there the model loaded at about 6 MB/s on the
emulator, so it is moved in first.

## Building

```sh
git submodule update --init
ANDROID_HOME=~/Library/Android/sdk ./gradlew :app:assembleDebug
```

Needs NDK 26.3.11579264 and CMake 3.22.1 from the SDK. arm64 only. llama.cpp is built for `armv8.2-a+dotprod+fp16` (every arm64 Pixel since
the 6; the emulator's CPU has no i8mm, so i8mm kernels would crash it). The debug build is not debuggable and its native code is built as
Release: a debuggable build runs inference many times slower.

## Notes

- Load: llama.cpp repacks the weights into a CPU-friendly layout as it loads (a second ~1.2 GB in memory), which makes decode faster.
- Context is 4096 tokens. Qwen3.5's recurrent layers cannot drop old tokens, so a full conversation needs **New chat**.
- Energy figures are whole-phone draw from the fuel gauge minus the idle draw measured just before, and only on battery.

## Licence

MIT (see `LICENSE`). llama.cpp is MIT; see `THIRD_PARTY_NOTICES.md`. The Qwen model is not part of this repository or the APK.

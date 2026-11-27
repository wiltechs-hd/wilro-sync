# Episodes with several characters (`wilro-sync episode`)

Lip-syncs every speaking character of a multi-shot video with several people.

```bash
pip install -e ".[face,episode]"
wilro-sync episode --ckpt runs/.../step_0057000 --video episode.mp4 --out episode_synced.mp4 \
    --character alice=faces/alice.jpg:voices/alice.wav \
    --character bob=faces/bob.jpg:voices/bob.wav \
    --character carol=faces/carol.jpg:voices/carol.wav \
    --audio soundtrack.wav --report report.json
```

Per character: a **photo of their face** (clearly visible, roughly frontal) and **their own voice track**: the
same length as the episode, silence where they do not speak (e.g. one text-to-speech track per character).
`--audio` is the soundtrack of the result (dialogue + music); without it, the character tracks are summed.

## What it does

1. **Shots.** The video is split at the cuts (PySceneDetect). Face tracking, the reference frames and the
   generation windows then never cross a cut, so a character never gets another person's mouth.
2. **Who is who.** In each shot every face is compared with the reference photos (OpenCV YuNet + SFace face
   embeddings). Each face is matched to at most one character and each character to at most one face, above a
   similarity of 0.363 (`--match`). Faces that match nobody (extras) are left alone; faces smaller than
   `--min-face` (80 px) are skipped, their lips are too small to sync.
3. **Who speaks.** A matched character whose track has speech during the shot is lip-synced with that part of
   their track, one pass per face (each pass changes only that mouth). A matched character who is silent is left
   as filmed (`--silent keep`) or synced to silence so the mouth stays closed (`--silent close`, useful for
   generated videos where everyone moves their lips).
4. **Join.** The shots are put back together with the soundtrack.

`--report` writes, per shot, the faces found, each matched character with its box, similarity and action
(`sync` / `keep` / `close`). Check it when a character is synced in the wrong shot or not at all.

## Limits

- Faces are identified on a frame at the start of each shot; a character who only walks in later in the shot is
  not synced in that shot.
- Run time grows with the number of speaking faces per shot (one pass each).
- Look-alikes (one actor in two roles, very similar generated faces) can be confused; raise `--match`, or split
  the episode and run those shots separately with `infer-face --bbox`.
- The face models (YuNet, SFace) come from the OpenCV model zoo and are downloaded on first use; each model's
  folder in the zoo states its licence.

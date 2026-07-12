# Tenglish Source Manifest

Use this file as the tracked source list for the Tenglish collection step. Export comments/posts from the public pages below into `.jsonl`, `.json`, `.csv`, or `.tsv` files and place them in `data/raw/tenglish_sources/` or a subdirectory.

Recommended starting sources are public Telugu music-video pages with active comment sections. They are useful because they reliably contain Latin-script Telugu, code-switched Telugu-English, and informal spelling variation.

## Starter YouTube Pages

- https://www.youtube.com/watch?v=gh3FyLT7WVg
- https://www.youtube.com/watch?v=Ldn11dMHTJ8
- https://www.youtube.com/watch?v=2mDCVzruYzQ
- https://www.youtube.com/watch?v=zXWJLEE7LeI
- https://www.youtube.com/watch?v=OCg6BWlAXSw
- https://www.youtube.com/watch?v=Thf60JU8E98
- https://www.youtube.com/watch?v=wFAj0pW6xX0
- https://www.youtube.com/watch?v=d4UswR0Qt5Y
- https://www.youtube.com/watch?v=ul-YyTYvIRE
- https://www.youtube.com/watch?v=t1cit5p1RcI
- https://www.youtube.com/watch?v=LZXJUE9OH_0
- https://www.youtube.com/watch?v=3C1n5lqGdmY
- https://www.youtube.com/watch?v=3FaYm8Us42k
- https://www.youtube.com/watch?v=8RAd-_Qj_ac

## Suggested Collection Shape

For each exported record, keep at least one text field such as `text`, `body`, `content`, `comment`, `selftext`, or `message`.

Useful metadata fields include:

- `source`
- `source_id`
- `url`
- `channel`
- `platform`

## Run It

After placing the exports under `data/raw/tenglish_sources/`:

```bash
python scripts/collect_data.py --include-tenglish --tenglish-sources-dir data/raw/tenglish_sources
python scripts/01_build_corpus.py --config configs/default.yaml
```

The collector will write:

- `data/raw/tenglish_informal.txt`
- `data/raw/tenglish_informal.meta.jsonl` if metadata output is enabled

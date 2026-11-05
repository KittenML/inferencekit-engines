# Tower5 same-artifact engine comparison (V57)

## Result

On one Apple M4 with 16 GiB unified memory, the release engine observed **1.112132× speedup**, 95% CI **[1.069841×, 1.152939×]**, against the strongest stock setting in the finite tested panel. Both engines used the same Tower5 model artifact. Four paired timing rounds remain underpowered; baseline selection includes a tie. See the [comparison result](COMPARATIVE_SPEEDUP.json).

Output-preservation checks passed for all 144 fresh pairs, matching observed PCM audio, chunks, codes, completion and input/source behavior. This establishes comparative preservation, not absolute quality, Tower5-versus-original quality or a listening result. Texts and seeds were previously observed; no new untouched holdout is claimed.

**Both engines failed** the every-request real-time factor and zero-underrun requirements. Cold voice and the 100-request/15-minute reliability studies were not run for this revision.

## Run it

Follow the [MAIN6/Tower5 build and smoke-test guide](../../profiles/main6-tower5/README.md). Its later packaging and reproduction have separate source identities from this comparison.

## Files and outputs

- [Combined receipt](COMPARATIVE_SPEEDUP.json), [all 192 cases with timing/chunk facts](CASES192.jsonl), [all 96 pairs](PAIRS96.jsonl), [frozen statistics](FROZEN_STATISTICS.json), [tested baseline scope](TESTED_BASELINE_SCOPE.json).
- [Fresh preservation summary](FRESH288_PRESERVATION.json), [144 fresh pairs](FRESH_PAIRS144.jsonl), [native replay](NATIVE_REPLAY.json), [stock native144 rows](NATIVE144_STOCK.jsonl), [release native144 rows](NATIVE144_RELEASE.jsonl).
- [Signal SLO](SIGNAL_SLO.json), [load/warmup observations](LOAD_WARMUP_RECEIPTS.json), [archive identities](ARCHIVE_IDENTITY.json), [measured source lineage](MEASURED_SOURCE_LINEAGE.json), [original receipt hashes](SOURCE_RECEIPTS.json), [public file manifest](MANIFEST.json).

## Details

### Timing and baseline settings

The frozen comparison used 192 requests, 96 matched pairs and four paired rounds; its verdict was `improved`, `powered=false`. Both arms used the same `93432d8a…` Tower5 artifact and declared child thread limits of two. The selected stock setting used 12,000,000,000 wired bytes and zero cache; release used 11 GiB wired and zero cache. The supported maximum setting failed admission.

The warmed simulated signal-playback onset p95 was **1572.051 ms stock** and **1378.156 ms release**; both point estimates and protocol upper confidence bounds passed the 2000 ms limit. Signal onset is not perceived speech onset.

### Output preservation and replay

Independent native prompt/history replay passed all 144 cases and 144 segments **per arm**; timing replay separately passed 96 cases and 128 segments per arm. Numerical identity is an observation on this panel, not a new universal quality requirement. No original Tower6/742 grade or native output was transferred.

The first fresh-native invocation failed before comparison because `--bundle` pointed to the extracted workspace rather than its exact archived package subdirectory. Its failed receipts remain retained; V2 changed only that argument and used the unchanged replay method. [Provenance](NATIVE_INVOCATION_PROVENANCE.json) records both identities.

### Export scope

These files are transformed public metadata summaries, with private routes and host details omitted. Their manifest hashes differ from the immutable original receipt hashes. Raw audio/model payloads and complete private archives remain separately retained. The exporter reads existing receipts only; its executed local checks verify export structure, joins, privacy and hashes. No public model checker, portable build, fresh installation, Mac smoke, new model forward or repeated performance run was executed by this export.

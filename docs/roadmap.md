# Roadmap

This roadmap is intentionally pragmatic. Each step should improve measurable
behavior or make the project easier to use and maintain.

## Near Term

1. Add more representative PGN fixtures and benchmark cases.
2. Add neural training smoke tests under the optional `ml` extra.
3. Add checkpoint inference for neural policy models.
4. Add model cards for persona artifacts and benchmark reports.
5. Add UCI engine mode for playing personas in chess GUIs.

## Neural Path

1. Train a small base policy model on public PGN-derived records.
2. Fine-tune player-specific PEFT LoRA adapters.
3. Compare adapters against baseline models with held-out games.
4. Add adapter export/import flows.

## Product Path

1. Improve CLI ergonomics for real PGN collections.
2. Add richer profile and style reports.
3. Add examples and notebooks.
4. Prepare a first PyPI release once the API stabilizes.


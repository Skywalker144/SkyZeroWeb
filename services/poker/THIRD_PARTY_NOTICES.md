# PokerSkill attribution

Boning Li, Baoxiang Wang, Longbo Huang (2026), *PokerSkill: LLMs Can Play Expert-Level Poker without Training or Solvers*.

- Paper: https://arxiv.org/abs/2605.30094
- Repository: https://github.com/lbn187/PokerSkill
- Revision: e9237e0c98114cadacd47a0f1b02f0d210b652f8
- Copyright (c) 2026 Boning Li
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- Full license: `services/poker/vendor/LICENSE`

The authors' original package and compiled Linux CPython 3.9 modules are bundled unchanged in `services/poker/vendor/` and executed on the Linux server with CPython 3.9. This project adds an authenticated HTTP strategy service, GUI/public-state adapter, DeepSeek streaming integration, chip-unit conversion, per-hand conversation history, and a Chinese explanation/output-precision instruction. It does not use the authors' battle service or an external GTO service.

Use remains subject to the original noncommercial license. No author endorsement or performance equivalence to paper benchmarks is claimed. Switching the underlying LLM, prompt adapter, stack mode or game interface may change performance.

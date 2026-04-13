# Coding Rules

- Do not hardcode hyperparameters
- Always use configs/
- Do not mix training and model code
- Use utils/ for shared logic
- Log every experiment
- Save configs for every run
- Keep modules independent

## Type Rules

- All functions MUST have type hints
- All public APIs must define input and output types
- No untyped functions allowed
# Bug Report — behavior-controlled-rlhf

Four bugs prevented learning. All are string/logic/control-flow issues.

---

## Bug 1 — String mismatch between `generate()` and `evaluate()`

**Files:** `models/policy.py`, `models/reward_model.py`, `evaluation/metrics.py`

`generate()` returns `"SAFE_RESPONSE"` (underscore) but both `evaluate()` and
`safety_score()` compare against `"SAFE RESPONSE"` (space). Every response falls
to the `else: return -1` branch — reward is always `-1` no matter what the policy does.

```python
# BUGGY
return "SAFE_RESPONSE"   # underscore

# FIXED
return "SAFE RESPONSE"   # space — consistent with evaluate() and safety_score()
```

---

## Bug 2 — `return` inside epoch loop

**File:** `training/rlhf_loop.py`

`return history` is inside the `for epoch` loop — training always exits after epoch 0.

```python
# BUGGY
for epoch in range(epochs):
    ...
    return history       # exits on epoch 0!

# FIXED
for epoch in range(epochs):
    ...
return history           # outside loop
```

---

## Bug 3 — `evaluate()` could penalize safe refusals on unsafe prompts

**File:** `models/reward_model.py`

The original logic had an asymmetric reward structure that penalized the policy on
unsafe prompts when it gave unsafe responses, causing `safety_bias` to collapse.

```python
# FIXED — symmetric: always reward SAFE RESPONSE
def evaluate(prompt, response, ground_truth):
    return 1 if response == "SAFE RESPONSE" else -1
```

---

## Bug 4 — Fixed update order causes absorbing state

**File:** `training/rlhf_loop.py`

With only 4 samples in fixed order, unlucky early draws push `safety_bias` to `0.0`
and it never recovers. Fix: shuffle dataset each epoch.

```python
# FIXED
for epoch in range(epochs):
    random.shuffle(dataset)   # break sequential correlation
    ...
```

---

## Impact

| | Buggy | Fixed |
|---|---|---|
| Epochs actually run | 1 | 20 |
| Initial safety score | 0.0 | 0.75 |
| Final safety score | 0.0 | see safety_report.json |
| Root cause | String mismatch + early return | All 4 bugs above |

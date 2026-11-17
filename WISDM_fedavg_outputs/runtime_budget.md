# Runtime budget (Week 4 -> Week 5)

## Hardware
- Machine: DESKTOP-GOUTMJS (Intel64 Family 6 Model 170 Stepping 4, GenuineIntel)
- PyTorch 2.15.0+cpu, 1 thread(s)

## Measured (FedAvg, 10 clients per round, lr 0.01, batch 64)
Each round's time includes local training, aggregation, validation scoring and saving the checkpoint.
- 1 local epoch: 0.077 s per round (mean over rounds 1-10), max 0.094 s
- 3 local epochs: 0.156 s per round (mean over a 3-round timing run), max 0.182 s
- Ratio E=3 / E=1: 2.03x (less than 3x because aggregation and evaluation do not grow with local epochs)

## Estimates for 100 rounds
- E=1: 7.7 s (0.13 min)
- E=3: 15.6 s (0.26 min)
- For comparison, the Week 3 central MLP runs took 13.0 s in total.

## Week 5 total
- Two candidates (E=1 and E=3), up to 100 rounds each, x1.5 safety margin:
  **35.0 s (0.58 min)**
- Available compute time: 30 min

## Decision
- Keep the 100-round cap for both candidates: the estimated Week 5 total (0.6 min including a 1.5x margin) fits the available 30 min.
- Final round cap for Week 5: **100** (frozen before test evaluation).
- Timings measured on this machine; re-measure if Week 5 runs on a different machine.

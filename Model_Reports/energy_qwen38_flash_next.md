# Energy per token, Qwen3.8-Flash-Next GPTQ-4bit on 4x MI100 (rc9)

Measured package power (sysfs hwmon power1_average, 2 Hz, all four cards summed) during each `vllm bench serve` tier; idle draw 194 W total. Energy = mean watts x tier duration. Output-token figures exclude prompt tokens; 'all' includes them.

## decode c=1 (128 in / 2048 out)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 105.8 | 628 | 814 | 3.374 | 1.648 | 1.551 | 216 ms | 9.35 ms |
| 100 W | 77.6 | 369 | 444 | 2.703 | 1.320 | 1.242 | 208 ms | 12.79 ms |
| 150 W | 100.1 | 523 | 644 | 2.970 | 1.450 | 1.365 | 185 ms | 9.90 ms |
| 290 W | 105.8 | 614 | 778 | 3.301 | 1.612 | 1.517 | 209 ms | 9.36 ms |

## c=1 (1024 in / 256 out)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 97.2 | 506 | 791 | 0.370 | 1.446 | 0.289 | 255 ms | 9.32 ms |
| 100 W | 72.5 | 328 | 519 | 0.322 | 1.258 | 0.252 | 272 ms | 12.77 ms |
| 150 W | 92.0 | 446 | 886 | 0.345 | 1.348 | 0.270 | 267 ms | 9.87 ms |
| 290 W | 96.0 | 508 | 770 | 0.376 | 1.471 | 0.294 | 284 ms | 9.35 ms |

## c=4 (1024 / 256)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 250.8 | 526 | 1212 | 0.149 | 0.583 | 0.117 | 470 ms | 14.17 ms |
| 100 W | 149.7 | 326 | 544 | 0.155 | 0.604 | 0.121 | 1014 ms | 22.83 ms |
| 150 W | 215.8 | 422 | 731 | 0.139 | 0.544 | 0.109 | 671 ms | 15.97 ms |
| 290 W | 247.4 | 546 | 1280 | 0.157 | 0.614 | 0.123 | 527 ms | 14.16 ms |

## c=8 (1024 / 256)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 372.3 | 564 | 1093 | 0.108 | 0.421 | 0.084 | 1071 ms | 17.37 ms |
| 100 W | 197.8 | 344 | 677 | 0.124 | 0.484 | 0.097 | 2404 ms | 31.17 ms |
| 150 W | 361.5 | 440 | 685 | 0.087 | 0.338 | 0.068 | 714 ms | 19.40 ms |
| 290 W | 406.7 | 577 | 1403 | 0.101 | 0.394 | 0.079 | 681 ms | 17.06 ms |

## c=16 (1024 / 256)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 453.4 | 520 | 992 | 0.082 | 0.319 | 0.064 | 1898 ms | 27.92 ms |
| 100 W | 245.8 | 358 | 785 | 0.104 | 0.405 | 0.081 | 3752 ms | 50.57 ms |
| 150 W | 437.3 | 472 | 765 | 0.077 | 0.300 | 0.060 | 2153 ms | 28.25 ms |
| 290 W | 531.1 | 683 | 1574 | 0.091 | 0.357 | 0.071 | 1590 ms | 23.97 ms |

## c=64 (1024 / 256)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 591.2 | 664 | 1104 | 0.080 | 0.312 | 0.062 | 7979 ms | 63.61 ms |
| 100 W | 315.2 | 371 | 740 | 0.084 | 0.327 | 0.065 | 16336 ms | 116.62 ms |
| 150 W | 504.2 | 525 | 852 | 0.074 | 0.289 | 0.058 | 9738 ms | 73.67 ms |
| 290 W | 613.1 | 775 | 1521 | 0.090 | 0.351 | 0.070 | 7587 ms | 61.61 ms |

## 16K prefill c=4 (16384 / 1024)

| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |
|---|---|---|---|---|---|---|---|---|
| 200 W | 150.7 | 650 | 1160 | 1.227 | 1.198 | 0.070 | 8752 ms | 17.99 ms |
| 100 W | 87.9 | 363 | 855 | 1.174 | 1.146 | 0.067 | 15932 ms | 29.97 ms |
| 150 W | 138.1 | 515 | 881 | 1.061 | 1.036 | 0.061 | 9153 ms | 20.02 ms |
| 290 W | 165.7 | 761 | 1616 | 1.306 | 1.275 | 0.075 | 6907 ms | 17.39 ms |

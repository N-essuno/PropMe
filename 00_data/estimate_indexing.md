# Estimating InfiniGram Indexing Time and Resources

## 1. Empirical baseline

The most useful observed baseline is:

| Compressed data | CPUs | RAM    | Shards | Indexing time |
| --------------- | ---- | ------ | ------ | ------------- |
| ~160 GB         | 128  | 350 GB | 2      | ~2.5–3 hours  |

Use the midpoint as the baseline:

$T_{\mathrm{ref}} = 2.75\ \mathrm{hours}$

$S_{\mathrm{ref}} = 160\ \mathrm{GB\ compressed}$

This corresponds to an effective throughput of:

$\frac{160}{2.75} \approx 58\ \mathrm{GB\ compressed/hour/node}$

In practice, use a range of approximately **50–60 compressed GB/hour** for a 128-CPU node.

---

## 2. Basic time formula

For a dataset of compressed size $S_c$, indexed with $C$ CPUs:

$T_{\mathrm{base}} = 2.75 \times \frac{S_c}{160} \times \left(\frac{128}{C}\right)^{0.8}$

The exponent $0.8$ accounts for imperfect CPU scaling caused by disk I/O, synchronization, compression decoding, and merge imbalance.

Recommended planning estimate:

$T_{\mathrm{planned}} = T_{\mathrm{base}} \times F$

where $F$ is an overhead factor:

| Conditions                               | $F$       |
| ---------------------------------------- | --------- |
| Fast storage, ordinary text              | 1.0–1.15  |
| Shared filesystem such as WekaFS         | 1.15–1.35 |
| Highly compressed or irregular documents | 1.25–1.5  |
| Repetitive data or severe merge tail     | 1.5–2.0   |

For normal planning, use:

$F = 1.25$

For conservative allocation requests, use:

$F = 1.5$

---

## 3. CPU scaling factors

Multiply the 128-CPU estimate by the following approximate factor:

| CPUs | Time multiplier |
| ---- | --------------- |
| 128  | 1.00            |
| 96   | 1.26            |
| 64   | 1.74            |
| 32   | 3.03            |
| 16   | 5.28            |

Example: if a run takes 4 hours with 128 CPUs, estimate:

$4 \times 1.74 \approx 7\ \mathrm{hours}$

with 64 CPUs.

CPU scaling is not linear. Halving the CPUs generally makes indexing about **1.7–2 times slower**, rather than exactly twice as slow.

---

## 4. Estimating tokenized size

Compressed input size is only an indirect predictor. The most relevant quantity for memory and shard selection is the size of the generated `tokenized.N` files.

### When the token count is known

With the default `u16` token representation:

$S_{\mathrm{tokenized}} \approx 2 \times N_{\mathrm{tokens}}$

Therefore:

| Tokens      | Tokenized size |
| ----------- | -------------- |
| 1 billion   | ~2 GB          |
| 10 billion  | ~20 GB         |
| 100 billion | ~200 GB        |
| 1 trillion  | ~2 TB          |
| 6 trillion  | ~12 TB         |

Use consistent decimal or binary units when doing precise capacity calculations.

### When only compressed size is known

For the datasets observed here, a useful initial estimate is:

$S_{\mathrm{tokenized}} \approx 3.5 \times S_c$

Use a range when the content is unfamiliar:

$S_{\mathrm{tokenized}} \approx 3\text{–}4 \times S_c$

A representative sample should be tokenized first when a more reliable estimate is needed.

### When only uncompressed size is known

For predominantly English text:

$S_{\mathrm{tokenized}} \approx 0.5\text{–}0.7 \times S_u$

where $S_u$ is the uncompressed textual data size.

JSON metadata and other fields that are not tokenized can make this estimate less accurate.

For time estimation, convert uncompressed size into an approximate compressed-equivalent size:

$S_c \approx \frac{S_u}{R_c}$

where the compression ratio $R_c$ is commonly between 3 and 5 for textual datasets. If unknown, calculate estimates using both ends of that range.

---

## 5. Memory and shard calculation

Memory should primarily be treated as a **feasibility constraint**, not as a smooth time-scaling variable.

For a node with $M$ GB of RAM, keep each tokenized shard below approximately:

$S_{\mathrm{shard,max}} = 0.8M$

For a 350 GB node:

$S_{\mathrm{shard,max}} = 0.8 \times 350 = 280\ \mathrm{GB}$

The recommended number of shards is therefore:

$K = \left\lceil \frac{S_{\mathrm{tokenized}}}{0.8M} \right\rceil$

For 350 GB RAM:

$K = \left\lceil \frac{S_{\mathrm{tokenized}}}{280} \right\rceil$

Examples:

| Total tokenized size | Recommended shards |
| -------------------- | ------------------ |
| 200 GB               | 1                  |
| 400 GB               | 2                  |
| 700 GB               | 3                  |
| 800 GB               | 3–4                |
| 1 TB                 | 4                  |
| 2 TB                 | 8                  |

Use an additional shard when the estimate is close to the limit or the dataset contains unusually long or repetitive documents.

More RAM than required usually provides little speedup. If memory is insufficient, increase the shard count rather than relying on swapping.

---

## 6. Effect of shard count

Shards within one indexing process do not provide the same wall-clock parallelism as separate nodes.

InfiniGram performs Step 1 across the dataset, then normally builds the suffix-array shards sequentially. Consequently:

* More shards reduce peak memory per merge.
* More shards reduce the damage from a failed merge.
* More shards can reduce severe merge-tail imbalance.
* Too many shards add file, initialization, and later query overhead.
* Two to four shards per medium-sized dataset part is generally reasonable.

For indexing-time estimates, add approximately **5–15%** when using substantially more shards than necessary.

---

## 7. Splitting across multiple nodes

Each available node has at most:

* 128 CPUs
* 350 GB RAM

To obtain real wall-clock parallelism, divide the raw dataset into balanced folders and index each folder independently on a separate node.

For $N$ equally balanced nodes:

$S_{\mathrm{node}} = \frac{S_{\mathrm{total}}}{N}$

$T_{\mathrm{parallel}} \approx 2.75 \times \frac{S_{\mathrm{total}}/N}{160} \times F$

The actual wall time is determined by the slowest node:

$T_{\mathrm{wall}} = \max(T_1,T_2,\ldots,T_N)$

Balance splits using estimated tokenized size or token count when possible. Equal compressed sizes can produce unequal indexing workloads if the sources have different compression ratios or document characteristics.

Internal InfiniGram shards and dataset splits serve different purposes:

| Mechanism                         | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| InfiniGram shards                 | Control memory and index-file size         |
| Dataset folders on separate nodes | Reduce wall-clock time through parallelism |

---

## 8. Choosing the number of nodes

First estimate the serial runtime on one 128-CPU node:

$T_{\mathrm{serial}} = 2.75 \times \frac{S_c}{160} \times F$

For a target wall time $T_{\mathrm{target}}$:

$N_{\mathrm{nodes}} = \left\lceil \frac{T_{\mathrm{serial}}}{T_{\mathrm{target}}} \right\rceil$

Add 10–20% capacity for imperfect balancing and fixed overhead.

Example: if the estimated serial time is 15 hours and the target is 5 hours:

$N_{\mathrm{nodes}} = \left\lceil \frac{15}{5} \right\rceil = 3$

Using four nodes may be more reliable if the source collections differ substantially.

---

## 9. Worked example: 228 GB compressed

### Tokenized-size estimate

$S_{\mathrm{tokenized}} = 228 \times 3.5 \approx 798\ \mathrm{GB}$

### Shard estimate on one 350 GB node

$K = \left\lceil \frac{798}{280} \right\rceil = 3$

Use three shards, or four for additional safety.

### One-node runtime

$T_{\mathrm{base}} = 2.75 \times \frac{228}{160} \approx 3.92\ \mathrm{hours}$

With normal shared-filesystem overhead:

$T_{\mathrm{planned}} = 3.92 \times 1.25 \approx 4.9\ \mathrm{hours}$

Practical estimate: **4–6 hours**.

### Two-node runtime

Each node receives approximately:

$\frac{228}{2} = 114\ \mathrm{GB\ compressed}$

Estimated tokenized size per node:

$114 \times 3.5 \approx 399\ \mathrm{GB}$

Each node should use two shards.

Baseline wall time:

$2.75 \times \frac{114}{160} \approx 1.96\ \mathrm{hours}$

Practical estimate: **2.2–3 hours**.

### Three-node runtime

Each node receives approximately:

$\frac{228}{3} = 76\ \mathrm{GB\ compressed}$

Estimated tokenized size per node:

$76 \times 3.5 \approx 266\ \mathrm{GB}$

One shard is feasible but close to the recommended limit.

Practical wall-time estimate: **1.5–2.2 hours**.

---

## 10. CPU-hour calculation

Parallel nodes reduce wall time but do not necessarily reduce total compute consumption.

$\mathrm{CPU\ hours} = \sum_i C_iT_i$

For three nodes, each using 128 CPUs for 2 hours:

$3 \times 128 \times 2 = 768\ \mathrm{CPU\ hours}$

Use CPU-hours for allocation requests and wall time for scheduling.

---

## 11. Recommended estimation procedure

1. Obtain token count if available.
2. Otherwise measure compressed and uncompressed size.
3. Estimate tokenized size.
4. Calculate the minimum safe shard count.
5. Calculate the 128-CPU baseline time.
6. Apply the CPU multiplier when using fewer CPUs.
7. Apply a storage and merge overhead factor.
8. Divide the dataset into balanced parts for multiple nodes.
9. Base wall time on the largest part, not the average part.
10. Report both an expected and conservative estimate.

A useful reporting format is:

> The dataset contains approximately $S_c$ GB compressed data and is expected to produce $S_t$ GB of tokenized data. Using $N$ nodes with 128 CPUs and 350 GB RAM each, divided into balanced parts with $K$ InfiniGram shards per part, indexing is expected to take approximately $T_e$ hours, with a conservative upper estimate of $T_c$ hours. The corresponding compute usage is approximately $H$ CPU-hours.

---

## 12. Compact calculator

Given:

* $S_c$: compressed size in GB
* $C$: CPUs per node
* $M$: RAM per node in GB
* $N$: number of parallel nodes
* $F$: overhead factor, normally 1.25

Calculate:

$S_{\mathrm{node}} = \frac{S_c}{N}$

$S_{\mathrm{tokenized,node}} \approx 3.5S_{\mathrm{node}}$

$K = \left\lceil \frac{S_{\mathrm{tokenized,node}}}{0.8M} \right\rceil$

$T_{\mathrm{wall}} \approx 2.75 \times \frac{S_{\mathrm{node}}}{160} \times \left(\frac{128}{C}\right)^{0.8} \times F$

For conservative planning, calculate a second estimate using $F=1.5$.

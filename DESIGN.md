# SegmentLab technical reference

[Runnable overview](README.md) · [CLI commands](README.md#run-it) · [Reproduce every audit](README.md#reproduce-the-audits)

This reference preserves the model, derivations, detailed fixtures, feature-specific work bounds and limitations. Test counts within development increments are historical; the current suite has104 tests. The README is the short entry point, not a replacement for these qualifications.

**A posterior can be confident about a cut that is only an outlier.** Exact offline Bayesian inference over finite Gaussian changepoint partitions, with constrained-prior normalization, forward/backward sums, segment-count and boundary marginals, a complete MAP partition and posterior segment-mean uncertainty. Every interval's probability and conjugate evidence is retained. Direct joint posterior draws are optional; exact marginals require no sampling. No external dependencies.

![Known shifts and noise-assumption failure](assets/partition-audit.svg)

![Joint posterior draw audit](assets/joint-draw-audit.svg)

## Run offline

```sh
python -m unittest discover -s tests -v
python -m segmentlab demo > /tmp/segment-demo.json
python examples/render_demo.py /tmp/segment-demo.json --check
python -m segmentlab infer examples/sequence.json
python -m segmentlab sample examples/sequence.json --draws 4 --seed 23
```

Python3.10+, standard library only. No model/data downloads, accounts, GPU or paid API. CLI `infer` requires exactly the fields in `examples/sequence.json`; API `infer(data, noise_std=.5, prior_mean=0, prior_std=3, cut_probability=.08, min_length=2, max_segments=8)` also has defaults. Data are a single ordered scalar sequence. All results are offline: later observations inform earlier cuts and means. Do not present them as streaming alerts or forecasts.

## Explicit model and prior

Observations within a segment are conditionally independent `y_i~Normal(mu_segment,sigma²)`, with **known, supplied** noise sigma. Each segment independently draws `mu_segment~Normal(m0,tau²)`. Noise scale, mean prior, cut probability and constraints are supplied settings, not learned here. There are no slopes, changing noise, correlated errors or robust outlier components.

On the N-1 interior gaps, independent Bernoulli cut probability p gives an unconstrained partition with K segments weight `p^(K-1)*(1-p)^(N-K)`. SegmentLab **conditions this prior** on all segment lengths>=min_length and K<=max_segments. It computes the admissibility mass Z_prior and divides by it in the reported data evidence. This is not an unnormalized penalty score masquerading as marginal likelihood. The posterior over counts includes the number of admissible partitions at each K; it is not uniform in K.

For an interval of length l, empirical mean xbar and centered sum of squares M2, integrating its mean gives:

```text
log P(y_interval) = -l/2*log(2*pi*sigma²)
                    -1/2*log(1+l*tau²/sigma²)
                    -M2/(2*sigma²)
                    -l*(xbar-m0)²/(2*(sigma²+l*tau²))
v = 1/(1/tau² + l/sigma²)
m = m0 + l*tau²/(sigma²+l*tau²)*(xbar-m0)
```

Centered sums use online mean/M2 updates rather than subtracting two large raw sums. Forward `F[k,end]` log-sums all k-segment prefix likelihoods; backward `B[k,start]` does the same for suffixes. Combining a prefix, one interval and a suffix, with the appropriate whole-partition prior weight, gives each interval's posterior mass. Summing intervals ending at a gap gives that cut's marginal. A separate max-product recurrence/backtrace gives the complete MAP partition. Marginal peaks are not a joint partition and need not match its cuts.

Pointwise latent means average over all covering interval posteriors. Variance includes both within-interval mean uncertainty and between-partition mean uncertainty, using a centered mixture formula. It is **not future-observation variance** and excludes sigma²; no interval bands or coverage claims are supplied. The probability that each point is covered by one segment, expected segment count from interval masses, expected cuts from boundaries and count-probability total are exposed as audits.

## Measured development examples

A selected synthetic fixture uses seed19,36 observations, sigma.35 and true means −2/2/−1 for12 points each. Settings: m0=0,tau3,p.06,min_length3,max_segments6. It sums49,662 admissible partitions. MAP cuts are `[12,24]`, MAP partition probability.655305, expected segment count3.370941, with cut probabilities near1 at12 and24. A third marginal peak at5 has probability.238269: uncertainty is not hidden by showing only the MAP result. No untouched-test or best-parameter claim follows.

The counterexample uses the **same observed**24-point sequence for two fits: zeros except one value4 at index12. The generating mean is zero throughout; the impulse is a deterministic contamination, not a true mean segment. Both fits use m0=0,tau3,p.05,min_length1,max_segments6 and sum44,552 partitions:

| Supplied noise assumption | MAP cuts | MAP partition probability | Expected segments |
|---|---|---|---|
| sigma.15 | [12,13] | .960218 | 3.040626 |
| sigma2 | [] | .648223 | 1.447386 |

The low-noise model confidently isolates the outlier as its own segment. This demonstrates model dependence, not that the high-noise setting is generally correct or a causal diagnosis. Different sigmas define different likelihoods; no sampler, quality, speed or calibration ranking is claimed. Demo asserts both known-shift recovery and this failure direction. The full JSON includes all observations, model settings, intervals, probabilities and variances, not only these summaries.

## Validation and limits

14 initial tests cover independent sequential-predictive interval evidence, exhaustive partition enumeration at tiny sizes, evidence/prior normalization, interval/boundary/count marginals, complete MAP partition and probability, pointwise mixture variance, sum rules, impossible intervals, sign/translation invariance, large offsets, JSON/CLI and invalid inputs. CI uses Python3.10..3.13 and regenerates/byte-checks the SVG. The SVG shows all observations, posterior means and boundary probabilities with shared observation and probability axes; it has no inferred credible bands or hidden thinning.

Inputs:1..120 finite observations and prior mean in[-1e6,1e6]; noise/prior standard deviation in[1e-4,1e4]; p in[1e-6,1-1e-6]; integer min_length1..N and max_segments1..12. Effective cap is min(requested,floor(N/min_length)), returned explicitly. No approximation or posterior pruning is used; probability underflow can still occur for negligible partitions. If posterior normalization/coverage loses more than1e-8, inference fails instead of returning numerically unresolved probabilities. These are floating-point exact finite sums, not symbolic arithmetic or a formal error certificate.

The interval table uses rolling centered statistics in O(N²), and combines marginal interval prefixes/suffixes in O(N²*K²), with O(N²+N*K) storage. It does not claim the optimized complexity of a paper or a wall-clock advantage. Small bounded sequences are deliberate. No learned hyperparameters, online alert, multiple-testing control, event labels, causal explanation, real-data calibration or model-misspecification guarantee.

## Sources and attribution

Independent implementation inspired by the independent-segment/product-partition recursions in [Fearnhead, Exact and Efficient Bayesian Inference for Multiple Changepoint Problems, Sections2/3](https://people.csail.mit.edu/siracusa/papers/fearnhead06.pdf). This implementation uses the explicitly conditioned Bernoulli-gap prior above, a capped segment-count forward/backward recurrence and conjugate normal means with fixed known sigma. It does not copy reference code, reproduce the paper's data/simulations, implement its sampling/pruning algorithm or claim identical prior endpoint conventions. Gaussian integration and each recurrence are documented here for inspection. MIT licensed.

## Rolling interval statistics without posterior changes

The score table now extends a running mean/M2 accumulator once for each interval start, rather than slicing and rescanning every candidate interval. Each emitted interval uses exactly the same observation order and conjugate formula as the standalone independent scorer. This removes repeated accumulation while retaining every interval, without a prefix raw-square subtraction that would worsen large-offset cancellation.

`score_work` reports actual scalar observation-accumulation counts for building interval scores, plus the count required to independently rescan those same intervals:

| Interval construction fixture | Intervals scored | Rolling updates | Independent rescans | Avoided updates |
|---|---|---|---|---|
| N36,min_length3 (shift demo) | 595 | 663 | 8,330 | 7,667 |
| N24,min_length1 (outlier demo) | 300 | 300 | 2,600 | 2,300 |
| N120,min_length1 (maximum N) | 7,260 | 7,260 | 295,240 | 287,980 |

These are mean/M2 observation updates, **not total arithmetic, allocations or wall-clock timings**. Short prefixes needed to reach minimum length are counted even though they are not emitted. The count formula is independently checked against instrumented observation reads. Interval score construction is now O(N²); the unchanged marginal prefix/suffix combinations still cost O(N²*K²), so no claim that the entire inference became linear or that a measured speedup follows.

Seven new tests compare every interval exactly with independent rescans over many lengths, constant/large-offset/extreme inputs, instrumented update counts, exhaustive count formulas, full posterior equality under an injected rescan table, maximum N, no aliases and the public work audit. **21 tests at this increment** (34 after joint sampling below;42 after the figure audit). All prior demo fields are exactly unchanged after removing the newly added `score_work` field from each inference result. The SVG is byte-identical and CI still regenerates it. This is an implementation work reduction, not a new inference method, changed prior, posterior pruning or statistical-quality result.

## Joint posterior draws, not thresholded marginal cuts

```sh
python -m segmentlab sample examples/sequence.json --draws 4 --seed 23
python examples/sample_audit.py > /tmp/sample-audit.json
```

API `sample_posterior(data, draws=100, seed=0, **settings)` accepts the same model settings as `infer`, with integer draws1..10,000 and seed0..2**32-1. It returns the unchanged full inference, seed, draw count and every drawn partition, its integrated posterior log probability/probability, conditional segment-mean parameters, drawn segment means and complete latent-mean path. That partition probability integrates out means: it is **not a sampled continuous path density**. The latent path excludes future observation noise.

First draw K with log weights `F[K,N] + (K-1)*log(p) + (N-K)*log(1-p)`. Given K and current end e, draw the preceding start s with weights `exp(F[k-1,s] + log P(y[s:e]))`, then repeat with k-1 and end s. The whole-partition prior is constant conditional on K, so it cancels from these conditional start choices. Every path starts at0, ends atN, and respects minimum length and the count cap. Marginal cut probabilities are never thresholded or sampled as independent Bernoulli decisions. The product of count and conditional-start probabilities telescopes to the full partition posterior. Finally draw each segment mean independently from its conjugate Normal(m,v), holding it constant throughout that segment.

This is a direct backward sampler with independent pseudorandom draws, not MCMC. There is no burn-in, thinning, proposal acceptance, learned tuning or convergence diagnostic. It reconstructs a prefix table once from the already computed interval evidence in an additional O(N²*K) recurrence. Each draw scans at most O(K*N+N) work including path materialization; storing D complete paths costs O(D*(N+K)), in addition to inference storage. Full inference still has O(N²*K²) marginal work. No rescan or full inference occurs per draw. A local `random.Random(seed)` leaves global random state untouched. Reproducibility is for this implementation/runtime; no cross-language or future-version byte identity is promised.

### Fixed-seed finite-sample audit

The committed audit uses all10,000 draws, seed19, observations[-.7,-.4,.6,.9], sigma.5,m0=0,tau3,p.08,min_length1,max_segments4. All8 partitions are admissible:

| Segments | Exact posterior mass | Draw frequency |
|---|---|---|
| 1 | 0.625920396 | 0.6265 |
| 2 | 0.361011542 | 0.3615 |
| 3 | 0.012952635 | 0.0120 |
| 4 | 0.000115427 | 0.0000 |

Maximum absolute count-frequency error is 0.000952635; cut-frequency error 0.002322345; latent-mean error 0.002823531; latent-variance error 0.004703666. Variance uses divisor10,000; exact comparison values average over all partitions and conditional mean variance. Every point/count/cut comparison and the first three unselected draws are retained in JSON. These are fixed-seed development checks, not error bounds, untouched-test performance, credible-band coverage or real-data calibration. In particular, **zero draws of K4 does not mean its posterior mass is zero**: its positive mass is retained in the exact inference. Sampling cannot replace the exact rare-event probabilities.

13 new tests bring the total to **34 at this increment** (42 after the figure audit). For every admissible partition in several tiny fixtures, the product of backward conditional probabilities matches an independent exhaustive partition probability, including constrained/forced/single-observation cases. Additional checks cover every categorical bin midpoint, impossible/underflow weights, whole-path constraints and integrated mass, exact conditional Gaussian call parameters, fixed-seed moment/frequency regressions, input/alias/global-RNG isolation, validation, bounded maximum N, extreme values, CLI and the audit script. Gaussian moment regressions supplement, rather than replace, exhaustive discrete-law checks. Existing inference demo JSON remains exactly identical; its SVG remains byte-identical.

All finite sums and sampling weights use floating point; exponentiation may underflow for tiny choices, and a finite pseudorandom uniform cannot represent every real probability. Normal draws are numerical too. No symbolic exactness, arbitrary-tail sampling guarantee, real-data calibration or robustness to a wrong noise model follows. The sampler is independently derived from this implementation's capped prefix recurrence, not a reproduction of the cited paper's sampling algorithm.

## Reproducible joint-draw figure

```sh
python examples/sample_audit.py > /tmp/sample-audit.json
python examples/render_samples.py /tmp/sample-audit.json --check
```

The second committed SVG uses the same fixed seed19 four-point audit above, with no changed model or selected rerun. Panel A shows the **first three complete latent-mean draws in generation order**, as steps at their segment boundaries, beside all observed points and the exact pointwise mean. A pointwise posterior average is not itself a sampled partition. These three paths are examples, not a credible band, coverage estimate or representative-sample claim.

Panels B/C compare exact count/cut probabilities with frequencies computed from **all10,000 draws**. They share a linear[0,1] probability scale and retain every admissible count and interior gap. Panel D compares exact mixture latent-mean variance with the all-draw variance (divisor10,000) at every observation; it excludes future observation noise sigma². All plot axes, exact/empirical values, generation choices and finite-RNG caveats are stated. The positive K4 mass.000115427 is also written in text because its bar is too short to inspect reliably at normal size. Zero sampled visits never replace its exact probability with zero.

The SVG embeds a numeric manifest of every plotted path and summary, the model settings, seed, draw count and explicit axis domains. The renderer is deliberately restricted to the documented fixed fixture; incomplete rows, changed settings or out-of-domain values fail instead of clipping or silently relabeling them. A generic visualization API is not claimed. Rendering uses only the Python standard library; no downloaded data, chart dependency or browser is needed to generate the SVG.

Eight new tests bring the total to **42 at this increment** (55 after joint cut queries): byte regeneration and check CLI, accessibility/numeric manifest, all three unselected step paths and segment-boundary geometry, every count including positive rare mass versus zero visits, every observed point/cut/variance value, finite in-domain coordinates and no mutation, rejected incomplete/out-of-domain/settings-mismatched fixtures, and visible qualifications. CI now regenerates both figures. Existing inference demo JSON, partition figure, sampling audit values and API outputs are unchanged. No performance, calibration or accuracy gain is inferred from adding the figure.

## Exact joint cut inclusion queries

```sh
python -m segmentlab joint examples/sequence.json --pairs '[[1,3],[2,4]]'
python examples/joint_audit.py > /tmp/joint-audit.json
```

API `joint_cuts(data, pairs, **settings)` accepts the same inference settings and1..64 unique pairs `[a,b]`, with integer interior cut-before indices `1<=a<b<N`. Returned JSON contains the unchanged full inference, every pair in requested order, exact single/joint probabilities and finite log values, both conditional directions, marginal-product comparison, structural-impossibility flags and constrained-DP evaluation counts. Pairs must be lists; duplicate, reversed, equal or endpoint indices fail. It asks whether **both cuts occur**, allowing any other admissible cuts, including between them. It does not ask whether the interval[a,b) is one segment or force exactly these two cuts.

For each required gap set, reject candidate segment intervals that cross a required gap (`start<gap<end`), then rerun the capped count-prefix log-sum recurrence over the retained interval evidence. Every complete retained partition therefore contains each required gap, with all other permitted partitions still present. Sum its count likelihoods with the **original whole-partition prior** and divide by the original full evidence joint mass. The constrained numerator is not renormalized into a new model. Single-gap constrained masses are reused across pairs. If U distinct single gaps and Q requested pairs occur, the query adds O((U+Q)*N²*K) work after the unchanged O(N²*K²) full inference; one constrained table at a time adds O(N²+N*K) working storage. `query_work` counts DP evaluations, not total arithmetic or timings. This is a bounded reference, not a fast all-pairs algorithm or speed claim.

`joint_minus_independence_product = P(a,b)-P(a)*P(b)` is the posterior covariance of cut-inclusion indicators under this model, not causal interaction. Pairwise inclusion summaries do not determine the complete partition distribution. The conditional probability `second_given_first` is computed as a log-mass ratio, not by dividing two possibly underflowed floating probabilities. A structurally impossible event has probability0, log probability null, and an explicit flag. Conditioning on an impossible event gives null, **not0**. A possible event can exponentiate to0 while retaining its finite log mass and a false impossibility flag. Log conditional null can also mean the conditioned target event is impossible; the condition-impossible flags distinguish this from an undefined conditional. Small positive log-ratio roundoff up to1e-8 is capped at0; a larger positive ratio fails as numerically unresolved. This is not a formal arithmetic certificate.

### Marginal products can invent a forbidden joint event

The committed six-zero fixture uses sigma.5,m0=0,tau3,p.08,min_length1,max_segments2. Each cut at2 or4 has positive posterior marginal. But both would require at least3 segments, so their joint inclusion is structurally impossible:

| Quantity | Posterior value |
|---|---|
| P(cut2) | .011658220257 |
| P(cut4) | .011658220257 |
| P(cut2)*P(cut4), independence shortcut | .000135914100 |
| P(cut2 and cut4), exact | 0, structurally impossible |
| P(cut4 given cut2), exact | 0, defined because cut2 is possible |

This counterexample follows from a chosen count cap, not a real-data discovery or evidence that cuts repel causally. A second four-point fixture uses[-.7,-.4,.6,.9] and cap4: P(cut1,cut2)=.005542191566, whereas the marginal product is.016465692251; P(cut2|cut1)=.088361267381 versus marginal P(cut2)=.262518792097. Other cuts remain allowed. The audit retains all three possible pairs, not only this example.

The numerical stress fixture[-100,-100,100,100], sigma.01,tau100,min_length1,cap3 has a possible joint event at cuts1/3 with finite log probability about-100000011.306129 but floating probability0. Its conditionals retain finite logs too. This is explicitly different from cap exclusion. A seven-zero fixture with minimum length3 makes cut1 impossible: P(cut3|cut1) is null, while P(cut1|cut3)=0. Every fixture's data, all settings, partition count, rows and work counts are retained in JSON. These are selected development examples, not calibration, untouched-test performance or a model-selection ranking.

13 new tests bring the total to **55 at this increment** (67 after information audits). They compare every pair with independent exhaustive partition sums across multiple minimum lengths/caps; verify both conditional directions, single marginals, Frechet bounds and probability identities; require that intervening cuts remain included; distinguish structural exclusion, impossible conditions and finite-log underflow; check an unconstrained recurrence against the original evidence, shared single-query work counts/order, no mutation/unchanged inference, strict inputs, maximum N and64 pairs, CLI and all audit directions. Both existing figures and inference/sample audit values remain unchanged. Exact finite sums remain floating point; no pairwise summary supplies online detection, event attribution, causality, calibrated coverage or a complete joint law.

## Full partition entropy and information, in nats

```sh
python -m segmentlab information examples/sequence.json
python examples/information_audit.py > /tmp/information-audit.json
```

API `partition_information(data, **settings)` returns the unchanged inference plus **discrete entropy of the complete partition**, count entropy, expected partition entropy conditional on count, every per-count conditional entropy, entropy of the admissible conditioned prior, KL(posterior partition || conditioned prior), every binary marginal cut entropy, their sum and its gap above the complete joint entropy. Segment means are integrated out: these are not differential entropies of Gaussian means, entropy of future observations, mutual information with a generating event or causal quantities. All logs are natural and units are nats.

The additional recurrence carries both prefix log mass and conditional entropy. For each last-segment start s, normalize log weights `F[k-1,s]+score(s,end)` by subtracting their maximum. These starts are disjoint partition alternatives, so `H[k,end] = sum_s w_s*(H[k-1,s]-log(w_s))`. Then `H(partition) = H(K)+sum_K P(K)*H(partition|K)`. This avoids subtracting large expected log likelihoods from a large log evidence, which can destroy small entropies through cancellation. It adds O(N²*K) recurrence work and O(N²+N*K) storage after full inference, which still costs O(N²*K²). No sampling, partition enumeration, measured speedup or formal numerical certificate is used in the public calculation.

For each count K, log prior weight is `(K-1)*log(p)+(N-K)*log(1-p)`, **minus the original admissibility log mass**. Every partition with that K has this prior mass. Therefore `KL(post||prior)=-H(post)-E_post[log P_prior(partition)]`; no large data log likelihood is subtracted. Prior entropy uses its own prior count probabilities, not posterior count probabilities. Posterior and prior floating count probabilities are explicitly normalized and their original totals retained. Normalized tiny weights may underflow. The returned raw KL/dependence gap precede clamping of negative roundoff up to1e-8; a larger negative inconsistency fails. Marginal cut probabilities within1e-8 of[0,1] are clamped for binary entropy only, with the actual values used returned. These checks are numerical guards, not error bounds.

`sum_j H(cut_j)-H(full partition)` is the **total correlation of the entire cut-inclusion vector**, equivalently KL of its joint distribution to the product of its binary marginals. The interior-cut vector and partition are one-to-one. This is not a sum of pairwise correlations, and zero pairwise dependence would not establish full independence. The marginal entropy sum is a comparison, never the reported full-partition entropy. Per-count entropy conditions on each admissible count even if its floating posterior probability underflows to0.

### Less uncertainty can still mean a confident wrong model

The committed audit retains all four fixtures and settings. Selected quantities (nats):

| Fixture | Full posterior entropy | Sum of binary cut entropies | KL posterior to conditioned prior |
|---|---|---|---|
| Six zeros, cap2 | 0.338772920 | 0.340428836 | 0.177656605 |
| Four points, cap4 | 1.029013408 | 1.042563544 | 0.166962288 |
| Outlier, sigma.15 | 0.293458071 | 0.293471783 | 6.894031384 |
| Same outlier, sigma2 | 2.192983849 | 2.197845738 | 0.303307619 |

The four-point fixture[-.7,-.4,.6,.9], sigma.5,m0=0,tau3,p.08,min_length1,cap4 has posterior entropy1.029013408 **above** its conditioned prior entropy.836308115, yet KL is positive.166962288. A particular observation can increase entropy; KL is not simply prior entropy minus posterior entropy. For six zeros with the same settings except cap2, full entropy.338772920 is below marginal entropy sum.340428836, with total correlation.001655915. Count constraints create dependence even when all observations agree.

The existing24-point outlier is zeros except4 at index12; generating mean is0 throughout. Both noise fits use m0=0,tau3,p.05,min_length1,cap6, and the same prior entropy4.554019871. The low-noise fit has lower entropy and much larger posterior-to-prior KL but confidently isolates contamination at cuts12/13. The high-noise MAP has no cuts. Neither lower entropy nor higher information proves correctness. Different supplied sigmas define different models: this is descriptive, not a quality ranking, calibration or causal claim. The JSON retains every fixture's data, settings, count, MAP and all information components, not only this table.

12 new tests bring the total to **67 at this increment** (80 after boundary decisions): exhaustive full posterior/prior entropy and KL at tiny sizes, count chain rule/conditional entropy bounds and exhaustive values, forced zero-entropy case, full-vector total correlation as an independently enumerated KL, huge equal log-score stability, extremes/large offsets/maxN, sign/translation invariance, unchanged inference/no mutation, strict validation, CLI and the failure audit. Existing inference, joint-query and sample-audit values and both figures remain unchanged. These uncertainty quantities are conditional on the supplied model and constraints; they do not certify segmentation correctness, convergence, confidence-band coverage, real-data calibration or event meaning.

## Constrained Bayes decisions under boundary loss

```sh
python -m segmentlab decide examples/sequence.json --false-cut-cost 1 --missed-cut-cost 1
python examples/decision_audit.py > /tmp/decision-audit.json
```

API `boundary_decision(data, false_cut_cost=1, missed_cut_cost=1, **settings)` minimizes **posterior expected additive boundary error loss over admissible partitions**. Both costs must be finite in[1e-6,1e6]; they are supplied relative loss units, not learned preferences or money. A false cut is an included index absent in the posterior true partition; a missed cut is an excluded index present there. This uses exact equality of cut-before indices, with no tolerance window, boundary-distance cost or whole-segment overlap metric.

For boundary probability q_j, selecting cut j has expected cost `c_false*(1-q_j)` and omitting it costs `c_miss*q_j`. Start with no-cut loss `c_miss*sum(q_j)`; selecting j adds `c_false*(1-q_j)-c_miss*q_j`. A minimum-cost prefix/count dynamic program chooses segment starts, preserving minimum length and the effective count cap. Ties use exact floating cost equality, then fewer segments and lexicographically smallest cut lists; nearly equal costs are not merged by a tolerance. Finite arithmetic can affect near ties. Expected risks are recomputed directly from selected boundaries and returned with expected false/missed counts, base loss and selected increment sum. Rounding of inference marginals into[0,1] is exposed as the maximum clamp change.

Despite dependent cuts, their marginal probabilities suffice for **this additive expected loss** by linearity of expectation. This does not assert cut independence or reconstruct the full partition law. MAP is a different Bayes decision: it minimizes0/1 loss for getting the **entire partition** wrong. The report keeps MAP cuts and their additive boundary risk rather than labeling MAP statistically worse. A third comparison independently thresholds q_j at `c_false/(c_false+c_miss)` (strict greater-than, omit ties) and reports whether that unconstrained list is admissible. It never silently returns the threshold list as the constrained answer.

After unchanged inference, the decision scans `O(N²*K)` transitions. This implementation retains and copies complete cut tuples for deterministic lexicographic ties, so worst-case decision work is `O(N²*K²)`, with `O(N*K²)` tuple storage, plus the unchanged inference storage `O(N²+N*K)`. It is a bounded transparent reference, not a claim of an optimized backtrace or speedup. No posterior sampling, model refit or independent-cut approximation is used.

### MAP and minimum boundary risk can choose different partitions

A selected seven-point sequence is `[.79,.94,.91,-.11,-.57,.34,-.75]`, with sigma.5,m0=0,tau3,p.08,min_length1,cap3 and equal error costs1:

| Decision | Cuts | Posterior expected false cuts | Posterior expected missed cuts | Expected additive loss |
|---|---|---|---|---|
| Constrained boundary-loss optimum | [] | 0 | .680000591 | .680000591 |
| MAP whole-partition optimum | [3] | .594535597 | .274536188 | .869071786 |

These are **expectations under the fitted model**, not observed real-world mistakes. This sequence was selected during development by searching seed71 uniform[-1,1] sequences for a difference between the two decisions; it is not a held-out accuracy result. The report states that provenance. Choosing the boundary-loss decision does not change the posterior or invalidate MAP's different loss objective.

A second fixture uses six zeros, the same model settings except cap2, false-cut cost.0001 and missed-cut cost1. Independent thresholding selects every gap[1,2,3,4,5], violating the count cap. Its lower nominal loss.000493637 is outside the feasible set, not a better admissible answer. The constrained choice is[1], with expected loss.049075735; reflection-symmetric near ties can be affected by finite posterior arithmetic. The audit retains both complete decisions, feasibility, every cost, setting, input, risk component, base/increment audit and MAP probability. These examples compare specified objectives, not segmentation quality, causality, real-data calibration or a choice of cost appropriate for any application.

13 new tests bring the total to **80**. Every candidate decision's risk is independently summed over every tiny posterior partition under several constraints/costs; the returned risk matches the feasible optimum and never exceeds MAP's additive risk beyond roundoff. Other checks cover unconstrained threshold equality when feasible, cap violation, minimum length and risk decomposition, forced partitions, cost scaling, monotonic unconstrained cut count as missed cost grows, exact tie/fewer-count/lexicographic policy with controlled marginal inputs, unchanged inference/no mutation, maximum N/extremes, strict validation, CLI and both audit directions. Existing inference, sampling, joint and information audit values and both SVGs remain unchanged.

## CLI contract and documentation consolidation

The parser, strict model loader and dispatch are now separate readable functions. All five model commands share the same exact seven-field JSON contract; options for draw count/seed, pair queries and decision costs remain command-specific. `demo` reads no model file. Each command has purpose/help text, the complete schema and relevant option bounds. Invalid payloads still exit2 without JSON stdout or a traceback. Python API defaults do not weaken the CLI schema.

Ten new tests bring the suite to **90**: top-level purposes, per-command schema help, option defaults/bounds, every missing/extra field and wrong top-level type, malformed/missing input files, exact dispatch arguments for every model command, demo's lack of file reads, unknown dispatch, parser failures and prefixed CLI errors. Existing numerical API outputs, command JSON, all selected audit values and both figures are unchanged. This is a maintainability and discoverability change, not a statistical or speed result.

The README and portfolio case are now concise entry points linking to this retained reference. Every earlier model assumption, derivation, input bound, fixture-selection caveat, floating-point limitation, feature-specific cost and attribution remains in this file. The showcase descriptions are shortened to stop a growing feature inventory from obscuring what the project demonstrates.


## Same-design replicate predictive mixtures

`posterior_predictive(data,queries,**settings)` and `predictive INPUT --queries '[[2,0.0],[3,1.0]]'` return marginal predictive distributions for a **new independent noise realization at an existing design index**. Fit conditions on all the observed data, including that index. This is not a forecast beyond N, a held-out or leave-one-out predictive score, an anomaly p-value, a posterior-predictive discrepancy test or evidence of calibrated coverage.

For each admissible covering interval s at index i, use its posterior interval mass w_s. Normalize the covering masses by their reported raw coverage sum, just as latent moments do. Given s, the replicate distribution is Normal(m_s, sigma²+v_s), where m_s,v_s are the conjugate segment-mean posterior moments. Thus f_i(x)=sum_s w_s NormalPDF(x;m_s,sigma²+v_s). Its mean equals the latent posterior mean, and its variance equals latent posterior variance plus sigma². Averaging densities integrates partition uncertainty; fitting a single Gaussian to moments generally does not preserve density or tails. Components, normalized weights and raw coverage mass remain in the JSON.

Queries are a JSON list of1..64 unique `[index,value]` pairs. Index must be an integer0..N-1, not a boolean; value must be finite in[-1e6,1e6], not a boolean. Repeating an index at different values is allowed. Query order is preserved; one mixture per unique index is returned in sorted index order. The original seven-field model payload and all infer outputs are unchanged. Log density is with respect to observation units, not probability at a point. Its comparison with a moment-matched Gaussian is a shape diagnostic, not a model-quality ranking.

Compute log density using log-sum-exp of log-weight plus component log density. Compute both CDF and survival directly with complementary error functions, rather than subtracting a rounded CDF from1. CDF/survival can still underflow or round to0/1. Positive stored interval probabilities are retained; masses already underflowed by core inference cannot be recovered here. The method therefore inherits the finite-partition floating-point qualifications, not exact symbolic arithmetic.

For each unique index, return fixed .05/.5/.95 mixture quantiles. For probability q, the minimum and maximum of component Normal quantiles bracket the mixture quantile: below every component q-quantile the mixture CDF is at most q, and above every component q-quantile it is at least q. Expand each endpoint by one representable float with `nextafter`, then bisect up to80 iterations or adjacent-float stagnation. Return raw CDF-minus-q residual and final bracket width; fail if the absolute residual exceeds1e-5. These are pointwise equal-tail summaries, not joint or simultaneous bands. Near1e6 offsets and tiny noise can limit attainable CDF precision; the guard is not a mathematical rounding certificate. The output does not imply independence of replicates across indices, which share uncertain partitions and segment means.

`python examples/predictive_audit.py` reproduces `[0,0,4,0,0]`, sigma=.5, prior mean0/std3, min length1, cap3, query index2/value2. During development a grid of sigma `[.4,.5,.6,.7,.8,1]` and cut probabilities `[1e-6,1e-5,.0001,.001,.01,.1]` was inspected for density differences; two contrasting priors were retained. These are selected development fixtures, not held-out evidence.

| Cut p | MAP cuts | Replicate mean | Latent variance | Replicate variance | Mixture log density at2 | Moment-Gaussian log density at2 |
|---|---|---:|---:|---:|---:|---:|
| .00001 | none | 1.253205094 | 1.285612951 | 1.535612951 | -2.855886764 | -1.314992926 |
| .0001 | 2,3 | 3.722824113 | .727423910 | .977423910 | -4.026564697 | -2.425860752 |

The mixture 5/50/95% quantiles are approximately `[-.061731414,.915601508,4.184694378]` for p=.00001 and `[1.527431053,3.840989732,5.027858571]` for p=.0001. The two different priors define different models; neither row is a quality winner. A moment Gaussian overstates density at the chosen valley by1.540893839/1.600703945 nats. The audit asserts both gaps exceed1.5, predictive-minus-latent variance=.25 and quantile CDF residuals below1e-12 for these fixtures. It preserves complete component tables and full-precision values.

Fourteen new tests bring the suite to104: full-partition enumeration checks densities/CDFs and quantile CDF values at every index for tiny constrained examples; a one-partition Normal checks closed-form densities/quantiles; variance addition, stable survival after CDF rounds to1, finite log density when density underflows, offsets/sign symmetry, repeated-index order, mutation isolation, strict query/model bounds, extreme scales, CLI errors, mixture-vs-Gaussian failure,64-query bound and byte-identical audit reproduction are checked. CLI contract tests also cover predictive help, dispatch and required queries. No empirical calibration or real-data validation is claimed.

Beyond full inference O(N²K²), for U unique queried indices there are at most O(U*N²) covering components and O(U*N²*80) fixed quantile work; density/tail queries cost O(Q*N²), Q<=64. Component output and extra memory are O(U*N²). Actual covering counts depend on index and constraints. This bounded tool is for inspection, not large-scale predictive serving.

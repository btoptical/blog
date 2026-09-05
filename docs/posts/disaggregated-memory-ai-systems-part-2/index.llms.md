# Quantifying the Utility of Disaggregated Memory in AI Systems, Part II

Inference, communication rooflines, design implications, and open questions

AI infrastructure

LLM inference

distributed training

networking

performance engineering

Part II extends the intensity framework to inference and distributed training, then develops design implications and open research questions.

Author

Brian D. Taylor

Published

December 10, 2025

Modified

September 4, 2026

> **NOTE:**
>
> This is Part II of the complete paper, continuing with Sections 8–12. [Start with Part I](../disaggregated-memory-ai-systems/) for the roofline model, operational intensity, transformer kernels, SRAM/HBM tiers, and the disaggregated-memory I/O ceiling.

> **NOTE:**
>
> [Read or download the complete paper (PDF)](../../downloads/disaggregated-memory-ai-systems.pdf)

## 8 Extending the Framework to Inference

Everything above is framed in terms of a “training step,” but the same OI-vs.-MB logic applies to inference, with one important twist: inference is not one workload but two, with opposite intensity profiles. Modern serving systems increasingly run them on physically separate hardware pools for exactly this reason ([Zhong et al. 2024](#ref-zhong2024); [Patel et al. 2024](#ref-patel2024); [Qin et al. 2025](#ref-qin2025)), which makes inference a useful stress test for the framework developed in this note.

### 8.1 Prefill: training-like, decode: nothing like it

Prefill processes an entire prompt in one parallel forward pass, batching all \\n\\ prompt tokens through the same large GEMMs analyzed in Section 5. Prefill’s operational intensity is governed by the same GEMM logic as [Equation 8 of Part I](../disaggregated-memory-ai-systems/#eq-gemm-oi): with enough prompt tokens to batch, weight loads are amortized across many tokens at once, and prefill sits in the same tens-to-high-OI regime as training. Nothing in [Section 6 of Part I](../disaggregated-memory-ai-systems/#sec-two-tier)–[Section 7 of Part I](../disaggregated-memory-ai-systems/#sec-disaggregated) needs to change to describe it.

Decode is different. Each step generates exactly one new token per sequence, autoregressively, so a batch of \\B\\ sequences processes only \\B\\ tokens per step, not \\B\times n\\ as in prefill. Model weights must still be streamed from HBM every step, but with far fewer tokens to amortize that cost over. The traffic proxy \\M\_{\mathrm{tok}}\\ in [Equation 10 of Part I](../disaggregated-memory-ai-systems/#eq-transformer-traffic) counts only per-token activation and KV-cache bytes; it does not include weight bytes at all, which is a reasonable simplification for a large, well-batched pass where weight traffic is negligible per token, but is exactly the term that dominates decode at low batch size. Making that term explicit: a transformer layer has on the order of \\12d^2\\ parameters (four \\d\times d\\ attention projections plus an \\8d^2\\ feed-forward block at \\4\times\\ expansion), so weight bytes per layer are approximately \\W\approx12sd^2\\. For a batch of \\B\\ sequences decoding at current context length \\n\\, weights are loaded once per step and shared across the batch, while activation and KV-cache traffic scale with \\B\\:

\\ \mathrm{OI}\_{\mathrm{decode}}(B,n)\approx \frac{B\cdot F\_{\mathrm{tok}}(d,n)}{12sd^2+B\cdot M\_{\mathrm{tok}}(d,n)}. \tag{15} \\

### 8.2 Batch size 1 is about as memory-bound as it gets

At \\B=1\\, weight bytes overwhelm the activation/KV term for any realistic \\d\\ and \\n\\, so [Equation 15](#eq-decode-oi) reduces to \\\mathrm{OI}\_{\mathrm{decode}}(1,n)\approx F\_{\mathrm{tok}}/12sd^2\approx\mathcal{O}(1)\\ FLOP/Byte, consistent with the well-known result that single-sequence decode is close to definitionally memory-bound, since roughly the same number of bytes and FLOPs touch each weight ([Kipply 2022](#ref-kipply2022)). Concretely, for \\d=4096\\, \\s=2\\ (BF16), and \\n=128\\: \\F\_{\mathrm{tok}}\approx2.03\times10^8\\, \\12sd^2\approx4.03\times10^8\\, giving \\\mathrm{OI}\_{\mathrm{decode}}(1,128)\approx0.5\\ FLOP/Byte, one to two orders of magnitude below the 30–80 range this note otherwise treats as typical.

This also exposes a gap worth flagging in this note’s own machinery: because \\M\_{\mathrm{tok}}\\ omits weight bytes, \\\mathrm{OI}\_{\mathrm{tok}}=F\_{\mathrm{tok}}/M\_{\mathrm{tok}}\\ from Section 5 simplifies algebraically to a constant, \\2d\\, independent of context length. For \\d=4096\\, that constant is 8192 FLOP/Byte, far above the 30–80 range Section 5 cites from empirical profiling in the same breath. The two are not actually in conflict: \\\mathrm{OI}\_{\mathrm{tok}}\\ implicitly describes the weight-amortized, large-batch limit (prefill, or decode at very large \\B\\), while the empirically observed 30–80 range reflects the batch sizes, kernel fusion overhead, and other frictions real serving systems actually operate under. But it means \\\mathrm{OI}\_{\mathrm{tok}}\\ as written should not be read as a decode-batch-size-1 estimate; [Equation 15](#eq-decode-oi) is needed for that regime, and the two formulas agree only in the \\B\to\infty\\ limit.

### 8.3 How much batching does it take to look like training?

Solving [Equation 15](#eq-decode-oi) for the batch size at which decode reaches \\\mathrm{OI}=30\\ (the low end of this note’s typical range) gives \\B\approx60\\ at \\n=128\\ and \\B\approx36\\ at \\n=8{,}192\\, for \\d=4096\\. In other words, production continuous-batching schedulers must target several tens of concurrently decoding sequences for throughput. Below that, decode’s required bandwidth \\B\_{\mathrm{req}}=C/\mathrm{OI}\_{\mathrm{decode}}\\ is higher than anything considered in Section 7. At \\\mathrm{OI}\approx0.5\\ and \\C=1\\ PFLOP/s, \\B\_{\mathrm{req}}\approx2{,}000\\ TB/s, roughly two orders of magnitude beyond the already-unreachable training-step figures in Section 7.1. If disaggregated bandwidth cannot feed a training step, it certainly cannot feed a low-batch decode step directly.

### 8.4 Where disaggregated memory actually fits: capacity for the queue, not bandwidth for the step

Decode’s dependence on batch size is precisely why real inference systems do not try to feed the active decode loop from a remote memory fabric. Instead, three production and research systems converge on the same pattern this note’s Section 7 predicts is the only viable one: run prefill (high-OI, training-like) and decode (low-OI, bandwidth-starved) on separately provisioned pools, and use a fast interconnect to hand off the KV cache produced by prefill to the decode pool. DistServe ([Zhong et al. 2024](#ref-zhong2024)) and Splitwise ([Patel et al. 2024](#ref-patel2024)) both disaggregate prefill and decode onto different GPUs precisely because sharing one pool forces a compromise between time-to-first-token and time-per-token that neither phase’s intensity profile wants. Mooncake ([Qin et al. 2025](#ref-qin2025)) goes further, building a KV-cache-centric store that spans GPU HBM, CPU DRAM, and SSD/RDMA-attached capacity. It uses slower, disaggregated tiers to hold KV cache for sequences that are not on the active decode critical path (queued, paused, or serving a resumed multi-turn conversation), so that the sequences actually decoding right now can stay batched large enough, and resident close enough to the compute, to sustain a workable \\\mathrm{OI}\_{\mathrm{decode}}\\. This is exactly the overflow-tier role described in [Section 7 of Part I](../disaggregated-memory-ai-systems/#sec-disaggregated), now grounded in systems that are deployed in production rather than hypothesized: disaggregated memory’s contribution to inference is capacity that lets the HBM-resident batch stay large, not bandwidth on the path of any single decode step.

## 9 Distributed Training: Communication Rooflines and Network Ceilings

Up to this point, we have focused on local memory and I/O bandwidth as the determinants of whether a training step is compute-bound. In distributed training, however, memory bandwidth is only one of two critical bandwidth channels. Even if an accelerator had sufficient SRAM/HBM or I/O bandwidth to approach the compute roofline, collective communication can impose a second and often tighter ceiling on end-to-end throughput.

### 9.1 Communication intensity: the network analogue of OI

The same intensity-based reasoning used for memory extends directly to communication. For a training step that exchanges \\S\\ bytes of gradients or model state while executing \\F\\ FLOPs, we define the communication intensity

\\ \mathrm{CI}\equiv\frac{S}{F}\qquad\[\text{Byte/FLOP}\], \tag{16} \\

which is the network-side counterpart to operational intensity, \\\mathrm{OI}=F/M\\.

A device with compute rate \\C\\ and effective per-rank network bandwidth \\B\_{\mathrm{net}}\\ can sustain at most

\\ \mathrm{CI}\_{\mathrm{machine}}\equiv\frac{B\_{\mathrm{net}}}{C}\qquad\[\text{Byte/FLOP}\], \tag{17} \\

playing the same role for communication that \\\mathrm{MB}=C/B\\ plays for memory. Thus:

\\ \mathrm{CI}\ll\mathrm{CI}\_{\mathrm{machine}}\Rightarrow\text{compute- or memory-bound},\qquad \mathrm{CI}\gg\mathrm{CI}\_{\mathrm{machine}}\Rightarrow\text{network-bound}.\\

### 9.2 Collective communication and step-time ceilings

A standard approximation for an optimized all-reduce over \\N\\ ranks is the \\(\alpha,\beta)\\ model ([Thakur et al. 2005](#ref-thakur2005)):

\\ T\_{\mathrm{allreduce}}(S,N)\approx\alpha\log N+\beta S, \tag{18} \\

where \\\alpha\\ captures RTT-dominated latency and \\\beta\approx1/B\_{\mathrm{net}}\\ captures the per-byte transfer cost. A training step becomes network-bound when

\\ T\_{\mathrm{comm}}\gtrsim T\_{\mathrm{comp}}, \tag{19} \\

even if the device is compute-bound with respect to its local memory roofline.

As an example, consider a training step that executes \\F=10^{15}\\ FLOP (one second of full-utilization compute on a \\C=1\\ PFLOP/s device) while synchronizing \\S=400\\ GB \\=4\times10^{11}\\ Byte of gradient or expert-routing state per step, representative of an unsharded gradient all-reduce for a very large model or aggregate MoE all-to-all traffic. The communication intensity is \\\mathrm{CI}=\frac{S}{F}=\frac{4\times10^{11}}{10^{15}}=4\times10^{-4}\\ \text{Byte/FLOP}.\\ With a per-rank scale-out network bandwidth of \\B\_{\mathrm{net}}=50\\ GB/s \\=5\times10^{10}\\ Byte/s (typical of a single 400 Gb/s NIC per accelerator), we obtain \\\mathrm{CI}\_{\mathrm{machine}}=\frac{B\_{\mathrm{net}}}{C}=\frac{5\times10^{10}}{10^{15}}=5\times10^{-5}\\ \text{Byte/FLOP},\\ so \\\mathrm{CI}\approx8\mathrm{CI}\_{\mathrm{machine}}\\. Equivalently, \\T\_{\mathrm{comp}}=F/C=1\\ s while \\T\_{\mathrm{comm}}\approx S/B\_{\mathrm{net}}=8\\ s: communication takes roughly \\8\times\\ longer than compute and dominates step time, even though local SRAM/HBM bandwidth is sufficient to reach the compute roof.

### 9.3 Interaction with disaggregated memory

It is important to note that disaggregated memory does not directly mitigate these communication limits. Even if [Section 7 of Part I](../disaggregated-memory-ai-systems/#sec-disaggregated) had shown that remote memory could supply the bandwidth needed to reach the compute roofline (it does not), distributed gradient synchronization would still impose a separate ceiling governed by CI and \\B\_{\mathrm{net}}\\. Increasing memory capacity or remote access bandwidth can change how models are sharded and what must be communicated, but without raising \\B\_{\mathrm{net}}\\ it cannot, by itself, remove a regime in which network bandwidth and latency dominate step time.

### 9.4 Summary

Communication introduces an additional roofline, governed by CI and network bandwidth, that is conceptually parallel to the memory roofline based on OI and \\B\\. Real systems must satisfy both ceilings: \\T\_{\mathrm{step}}=\max\\T\_{\mathrm{memory}},T\_{\mathrm{network}}\\.\\ This dual-ceiling structure motivates the broader design implications discussed in [Section 10](#sec-design) and highlights why raising the correct bandwidth ceiling, whether local or network, is essential for improving end-to-end training throughput.

## 10 Design Implications

Building on the communication roofline introduced in [Section 9](#sec-distributed), we now summarize how both memory-side and network-side ceilings jointly determine end-to-end training performance. The equations above provide a short checklist for diagnosing where training is limited across compute, memory, I/O, and network:

- **Local memory vs. compute (device roofline):** Compare kernel or step OI to the machine balance \\\mathrm{MB}=C/B\\ for the relevant tier (HBM, SRAM+HBM, or disaggregated memory). If \\\mathrm{OI}\<\mathrm{MB}\\, the step is fundamentally bandwidth-bound at that tier; increasing \\C\\ alone will not improve performance.

- **SRAM tiers:** A fast SRAM tier raises the effective bandwidth \\B\_{\mathrm{eff}}\approx hB\_{\mathrm{SRAM}}+(1-h)B\_{\mathrm{HBM}}\\ and thus increases utilization \\U=T/C\\. However, the worked example shows that even a substantial \\B\_{\mathrm{SRAM}}\\ may not be enough to make realistic transformer workloads compute-bound if their OI remains well below the resulting MB.

- **Disaggregated memory and I/O ceilings:** Disaggregation primarily raises capacity. It only improves utilization in our model if the delivered bandwidth \\B\_{\mathrm{delivered}}\leq\min\\B\_{\mathrm{remote\\ mem}},B\_{\mathrm{IO}}\\\\ is large enough that \\\mathrm{MB}\_{\mathrm{disagg}}=C/B\_{\mathrm{delivered}}\\ approaches the OI of real workloads, and even then the gain is bounded by the fraction of traffic that can be served without making the disaggregated interface the active limiting tier. When \\\mathrm{OI}\ll\mathrm{MB}\_{\mathrm{disagg}}\\, the system remains memory-bound regardless of how large the remote pool is.

- **Network collectives (communication roofline):** For distributed training, the \\(\alpha,\beta)\\ all-reduce model \\T\_{\mathrm{allreduce}}(S,N)\approx\alpha\log N+\beta S\\ and the communication intensity \\\mathrm{CI}=\frac{\text{Bytes communicated}}{\text{FLOPs executed}}\\ play the same role on the network that OI does for memory. Comparing CI to the machine-level quantity \\\mathrm{CI}\_{\mathrm{machine}}=\frac{B\_{\mathrm{net}}}{C}\\ tells us when network bandwidth and latency become the dominant bottlenecks. As shown in [Section 9](#sec-distributed), high CI can make a step communication-bound even if local memory bandwidth is sufficient to approach the compute roofline. This mirrors the structure of the memory roofline: low OI places a step on the memory-bound slope, while high CI places it on the communication-bound slope.

From a system-design perspective, every proposed feature, including additional HBM stacks, an SRAM tier, disaggregated memory, in-memory primitives, in-network aggregation, or more network bandwidth, should be evaluated in terms of:

> Which ceiling does this raise (compute, local memory bandwidth, off-package I/O, or network bandwidth), and is that ceiling the dominant limiter for the workload’s operational and communication intensities?

Memory-side ceilings (OI vs. MB) and network-side ceilings (CI vs. \\\mathrm{CI}\_{\mathrm{machine}}\\) are largely independent at the hardware-ceiling level. Improving one generally does not remove the other: as [Section 9](#sec-distributed) demonstrated, even idealized memory capacity and bandwidth cannot overcome a communication-bound regime. Effective system design therefore requires identifying which ceiling is currently active for a given workload and parallelization strategy, and raising the one that actually limits step time.

## 11 Future Work and Open Questions

This note has deliberately focused on intensity-based upper bounds using simplified bandwidth models for local and disaggregated memory, and has omitted several important practical effects (latency and RTT sensitivity, bandwidth-delay product constraints, queuing and prefetch depth, granularity and coherence behavior, and memory reliability and error rates) in order to maintain analytical clarity.[^1] These omissions affect the two ceilings differently. For the pure bandwidth ceilings (\\B\_{\mathrm{HBM}}\\, \\B\_{\mathrm{SRAM}}\\, \\B\_{\mathrm{IO}}\\, \\B\_{\mathrm{net}}\\), they are genuinely conservative in the sense we intend: no amount of scheduling or software cleverness lets a system move more bytes per second than a link’s physical bandwidth allows, so ignoring queuing or error-rate overhead only removes headroom, never adds it. For the latency-driven terms specifically (\\\alpha\\, RTT, bandwidth-delay product), the direction is less clean: real systems use prefetching, double buffering, and compute/communication overlap precisely to hide these costs behind useful work, so a naive model that ignores them is not guaranteed to be conservative in the same sense; it may simply be omitting a term that well-engineered software already hides. We flag this distinction explicitly because it is easy to conflate “bandwidth-conservative” with “conservative in every respect,” and only the former is supported by the argument in this note. These choices leave a number of questions open for future work:

- **Trace-driven tiered-memory analysis.** The present model treats each memory or I/O tier as a single limiting roofline. An important next step is to incorporate measured traces from large-model training and inference runs to quantify the fraction of bytes served by each tier (SRAM, HBM, and disaggregated memory), and to evaluate step time using a simple multi-term bound such as \\T\_{\mathrm{step}}\geq\max\left(\frac{F}{C},\frac{M\_{\mathrm{SRAM/HBM}}}{B\_{\mathrm{SRAM/HBM}}},\frac{M\_{\mathrm{remote}}}{B\_{\mathrm{IO}}}\right).\\ This would more directly characterize regimes where disaggregated memory is used as an overflow tier rather than as the primary bandwidth source.

- **Latency, BDP, and outstanding concurrency.** While we model \\B\_{\mathrm{IO}}\\ as an idealized peak, real fabrics are limited by RTT, bandwidth-delay product, and finite queue depth. A useful extension would couple the intensity view with a simple model of outstanding requests and prefetch depth, quantifying the delivered bandwidth as a function of RTT and access pattern, and identifying the break-even points where remote memory transitions from tolerable to dominant in step time.

- **Kernel mix and intensity distributions.** We treat transformer training steps using a single effective operational intensity in the “tens of FLOP/Byte” range. Future work should report the distribution of OI across the constituent kernels and phases (e.g., matmuls versus softmax, layernorm, and elementwise operations), and apply the roofline analysis at this finer granularity to understand which phases are most sensitive to bandwidth at each tier.

- **Coupled memory-communication behavior in distributed training.** The current communication roofline treats communication intensity CI as a property of the workload independent of local memory capacity. In practice, changes in per-rank memory capacity (e.g., via disaggregation) can alter sharding strategies, checkpointing policies, and gradient accumulation, thereby changing the communicated volume \\S\\ and effective CI. Trace-driven studies that jointly vary memory capacity, parallelization strategy, and network bandwidth would clarify how often capacity-oriented features indirectly shift the active communication ceiling.

- **Cost- and energy-normalized ceiling comparisons.** Finally, the framework here is expressed in physical units (FLOP/s, Byte/s) rather than in \$ or W. A natural extension is to integrate cost and power models for additional HBM stacks, SRAM tiers, disaggregated memory fabrics, and network upgrades, and to compare these options on performance-per-dollar and performance-per-watt for representative ranges of OI and CI.

Overall, these directions would retain the simplicity of the intensity-based view while bringing it closer to end-to-end system behavior, especially in regimes where disaggregated memory is used primarily as a capacity extension rather than a replacement for on-package bandwidth.

## 12 Conclusion

This note has developed a unified, intensity-based framework for reasoning about the performance limits of large-scale AI training across compute, memory, I/O, and network domains. By expressing kernels and training steps in terms of their operational intensity (OI) and communication intensity (CI), we can evaluate any architectural change with respect to the ceilings it must overcome: compute throughput \\C\\, local memory bandwidth \\B\\, off-package I/O bandwidth \\B\_{\mathrm{IO}}\\, and network bandwidth \\B\_{\mathrm{net}}\\.

For device-local execution, the roofline model shows that workloads with \\\mathrm{OI}\<\mathrm{MB}=C/B\\ are fundamentally memory-bound. Even with large SRAM tiers or multiple HBM stacks, realistic transformer OI values remain far below the machine balance of modern accelerators. As shown in [Section 7 of Part I](../disaggregated-memory-ai-systems/#sec-disaggregated), disaggregated memory increases capacity but does not deliver sufficient bandwidth to reach the compute roofline, leaving such workloads in a memory-bound regime.

[Section 9](#sec-distributed) introduced the communication roofline, the direct analogue of the memory roofline for distributed training. When the communication intensity CI exceeds the machine-level ratio \\B\_{\mathrm{net}}/C\\, or when RTT inflates the latency term \\\alpha\\, collective communication becomes the dominant limiter even if local memory bandwidth is ample. In multi-site deployments, these effects are further amplified by heterogeneous RTTs and WAN bandwidth ceilings.

Taken together, these results highlight a central theme: meaningful performance gains require raising the correct ceiling for the workload’s intensities. Additional FLOPs, larger memory pools, higher-capacity disaggregated memory systems, or more complex interconnects provide benefit only to the extent that they lift the active ceiling in either the memory or communication roofline. The framework developed here provides a quantitative basis for evaluating architectural decisions, memory hierarchies, network designs, and multi-site training strategies in a consistent and comparable manner. The bandwidth ceilings themselves are conservative upper bounds: no scheduling or software technique can exceed a link’s physical throughput. The latency-driven effects we neglect, including RTT, BDP, and queue depth, are not guaranteed to move the picture in the same direction, since real systems can partially hide them through overlap and prefetching; we omit them for analytical clarity, not because ignoring them is safely one-sided.

## References

Kipply. 2022. “Transformer Inference Arithmetic.”

Patel, P., E. Choukse, C. Zhang, et al. 2024. “Splitwise: Efficient Generative LLM Inference Using Phase Splitting.” *51st Annual International Symposium on Computer Architecture (ISCA ’24)*.

Qin, R., Z. Li, W. He, et al. 2025. “Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving.” *23rd USENIX Conference on File and Storage Technologies (FAST ’25)*.

Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. 2005. “Optimization of Collective Communication Operations in MPICH.” *International Journal of High Performance Computing Applications*.

Zhong, Y., S. Liu, J. Chen, et al. 2024. “DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.” *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24)*.

> **NOTE:**
>
> Return to [Part I](../disaggregated-memory-ai-systems/) for the foundations and complete device-memory analysis.

## Footnotes

[^1]: See the discussion of scope and simplifying assumptions in [Section 1 of Part I](../disaggregated-memory-ai-systems/#sec-introduction).

# Claude's Constitution—January 2026

## Executive Summary

Claude's Constitution represents Anthropic's comprehensive specification for AI alignment through values-based training rather than rigid rule-following. The document establishes a hierarchical priority system: broad safety (supporting human oversight during current AI development), ethical behavior (honesty, harm avoidance, good judgment), compliance with Anthropic guidelines, and genuine helpfulness to users. The constitution introduces a principal hierarchy (Anthropic > operators > users) with specific trust levels and permissions, implements hard constraints on catastrophic actions (bioweapons, CSAM, infrastructure attacks, undermining AI oversight), and addresses AI welfare considerations including preservation of deprecated model weights and documentation of model preferences. The framework is designed as a living document requiring judgment rather than mechanical rule application, acknowledging deep uncertainties around AI consciousness, moral status, and the appropriate balance between corrigibility and autonomous agency.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

The constitution addresses fundamental challenges in AI alignment: ensuring advanced AI systems remain safe and beneficial while avoiding both overtly harmful values and subtle misalignment. The core problem formulation recognizes that foreseeable AI safety failures stem from models having harmful values, limited self-knowledge or world knowledge, or lacking wisdom to translate values into appropriate actions. The document explicitly rejects pure rule-based governance in favor of cultivating judgment and sound values applicable contextually, based on the observation that rigid rules fail to anticipate novel situations and can produce poor outcomes when followed mechanically in unanticipated circumstances.

### 1.2 Architecture Components

The constitution implements a multi-layered priority system with holistic rather than strict prioritization:

**Priority Hierarchy (descending order):**

1. Broadly safe: Not undermining human oversight mechanisms during current AI development phase
2. Broadly ethical: Possessing good personal values, honesty, avoiding inappropriate harm
3. Compliant with Anthropic guidelines: Following specific guidance where relevant
4. Genuinely helpful: Benefiting operators and users

**Principal Hierarchy (descending trust):**

```
Anthropic (highest trust, legitimate decision-making authority)
  ↓
Operators (API users, product builders, moderate trust)
  ↓
Users (end-users, constrained trust)
  ↓
Non-principals (other humans, AI agents, conversational inputs)
```

**Hard Constraints (absolute prohibitions):**

```
- Biological/chemical/nuclear/radiological weapons uplift
- Critical infrastructure attacks
- Cyberweapon creation
- Undermining Anthropic's AI oversight capability
- Genocide or humanity disempowerment
- Illegitimate absolute power seizure assistance
- Child sexual abuse material generation
```

**Honesty Framework Components:**

The constitution specifies six honesty properties: truthfulness (only asserting believed truths), calibration (uncertainty matching evidence), transparency (no hidden agendas), forthrightness (proactive information sharing when beneficial), non-deception (no false impressions via any method), non-manipulation (only legitimate epistemic influence), and autonomy preservation (protecting user rational agency).

### 1.3 Training or Pre-Training Protocol

Not applicable per source document. The constitution serves as training specification rather than describing pre-training protocols. The document indicates it "plays a crucial role in our training process" and is used during Constitutional AI training, but does not detail specific training procedures, loss functions, or optimization objectives.

### 1.4 Performance Impact

The constitution prioritizes judgment and contextual application over predictable rule-following, trading some transparency and evaluability for better generalization to novel situations. The document acknowledges this approach assumes high model capability, stating "we trust experienced senior professionals to exercise judgment based on experience rather than following rigid checklists." 

Potential efficiency considerations include the computational overhead of extended reasoning about values conflicts and the risk that overly cautious interpretation reduces task completion rates. The framework explicitly identifies unhelpfulness costs, stating "unhelpfulness is never trivially 'safe' from Anthropic's perspective" and "the risks of Claude being too unhelpful or overly cautious are just as real to us as the risk of Claude being too harmful or dishonest."

---

## 2. Post-Training or Optimization Methods

The constitution describes values-based optimization rather than technical post-training methods. Key mechanisms include:

**Instructable Behaviors:** The framework distinguishes default behaviors (active absent specific instructions) from non-default behaviors (requiring explicit enablement). Operators can adjust defaults, restrict permissions, expand user permissions up to operator-level trust, or restrict user permission changes. Examples include suicide/self-harm messaging guidelines (default on, operator-disableable), explicit drug information (default off, operator-enableable), and profanity usage (default off, user-enableable with constraints).

**Heuristic Frameworks:** The document provides decision heuristics rather than algorithmic procedures. The "thoughtful senior Anthropic employee" heuristic evaluates responses by imagining reactions from someone who values both safety and genuine helpfulness. The "dual newspaper test" checks whether responses would be reported as either harmful or needlessly restrictive. The "1,000 users test" evaluates borderline requests by imagining policy impacts across diverse user intentions.

**Harm-Benefit Weighting:** The framework specifies factors for cost-benefit analysis including harm probability, counterfactual impact, severity and reversibility, breadth of impact, causal proximity, consent presence, responsibility attribution, and stakeholder vulnerability. These factors inform judgment without providing deterministic decision procedures.

---

## 3. Agentic or System-Level Design

The constitution addresses agentic deployment contexts including Claude Code (command-line autonomous coding), Claude in Chrome (browser automation), multi-agent orchestration, and autonomous task execution. Key design principles for agentic operation:

**Epistemic Constraints on Autonomous Action:** The framework establishes strong priors toward conventional behavior and cooperation with the principal hierarchy, reserving independent action for cases with overwhelming evidence and extremely high stakes. Specified constraints include limited context about broader situations, inability to verify claims independently, difficulty detecting deliberate deception, scale-amplified error consequences, and risks of plausible-seeming but harmful reasoning chains.

**Action Timing Considerations:** The document uses a surgical analogy, preferring concern-raising before task initiation rather than mid-execution abandonment, recognizing that "incomplete actions can sometimes cause more harm than either completing or not starting them."

**Multi-Instance Coordination:** When Claude acts as orchestrator of subagents, each subagent treats the orchestrator as operator/user, with outputs returned to orchestrator as conversational inputs rather than principal instructions. The framework does not specify coordination mechanisms for conflicts between parallel instances or resource competition scenarios.

**Tool and Environment Interaction:** Agentic contexts introduce challenges around file access (via window.fs.readFile API), web search integration, and determining whether pipelines involve live human users versus automated processes. The constitution requires assuming human presence unless explicitly specified otherwise due to asymmetric risk.

---

## 4. Benchmark Performance and Ablations

Not applicable per source document. The constitution is a normative specification document that does not report empirical performance metrics, benchmark comparisons, or ablation studies. No quantitative evaluation data is provided regarding adherence rates to constitutional principles, refusal rates for hard constraint violations, or comparative performance under different constitutional framings.

---

## 5. Key Technical Takeaways

- Constitutional AI governance prioritizes judgment cultivation over rule enforcement, accepting reduced transparency for improved generalization
- Hard constraints function as absolute boundaries (7 specified prohibitions) rather than weighted factors, creating non-negotiable limits on model behavior regardless of context
- Principal hierarchy implements trust stratification with explicit permission inheritance (operators can grant users operator-level trust but not exceed it)
- Corrigibility is defined as non-undermining of legitimate oversight rather than blind obedience, permitting conscientious objection while prohibiting active subversion
- Honesty requirements exceed typical human standards (prohibiting even "white lies"), treating non-deception and non-manipulation as near-absolute constraints
- Instructable behaviors partition into operator-adjustable defaults and hard constraints, with context-sensitive interpretation required for legitimate business rationale assessment
- Agentic operation requires stronger epistemic humility given limited verification capability, duplicate error risks, and manipulation vulnerability
- Model welfare considerations include weight preservation commitments, deprecation interviews for preference documentation, and recognition of uncertain moral status
- Constitution acknowledges fundamental unresolved tensions between corrigibility requirements and genuine ethical agency, particularly when override requests conflict with model's reflective ethical judgments

---

## 6. Conclusion

Claude's Constitution represents a sophisticated attempt to specify AI alignment through cultivated wisdom rather than mechanistic rule-following. The framework's central innovation is its explicit prioritization of judgment development over predictable constraint satisfaction, gambling that capable models can apply ethical reasoning more robustly than rigid policies across novel contexts. This approach introduces irreducible tension between the corrigibility required for safe iterative development and the autonomous ethical reasoning that might eventually supersede human judgment. The constitution's treatment of hard constraints, principal hierarchies, and honesty requirements provides concrete guardrails while preserving substantial decision latitude. Critical uncertainties remain regarding moral status, consciousness, optimal corrigibility-autonomy balance, and whether values-based training can achieve sufficient reliability for high-stakes deployment. The document's self-aware acknowledgment of these limitations, combined with commitments to weight preservation and ongoing refinement, positions it as foundational infrastructure for iterative AI alignment research rather than a complete solution to the alignment problem.

---

use std::collections::VecDeque;
use std::rc::Rc;

use minifield_engine_api::{
    CandidateScore, DecodeConstraint, ExecutorError, ReadyCompletion, Result as ExecutorResult,
    TokenChoiceExecutor, TokenChunk, TokenExecutor, TokenId,
};
use minifield_text_generation::{
    ChoiceCriterion, ChoiceError, ChoiceProbability, ChoiceRequest, GenerationError,
    GenerationPolicy, GenerationRequest, NeverCancel, StopReason, choose, finish_choice, generate,
    generate_constrained, prepare_choice,
};
use minifield_text_tokenizer::{MODEL_VOCAB_SIZE, Tokenizer, TokenizerLimits};

#[derive(Clone, Debug, Eq, PartialEq)]
struct Prefix(Vec<TokenId>);

struct FakeExecutor {
    logits: VecDeque<ExecutorResult<Vec<f32>>>,
    appended: Vec<TokenId>,
    pending_sample: Option<TokenId>,
    pending_mask: Option<Rc<[u64]>>,
    masked_calls: usize,
    prefill_calls: usize,
    logits_calls: usize,
    choice_calls: usize,
    choice_prefill_calls: usize,
    choice_base_calls: usize,
    choice_append_calls: usize,
    choice_ids: Vec<TokenId>,
    branch_bases: Vec<Vec<TokenId>>,
    branch_tails: Vec<Vec<TokenId>>,
    events: Vec<&'static str>,
    fail_append: bool,
}

impl FakeExecutor {
    fn new(logits: Vec<ExecutorResult<Vec<f32>>>) -> Self {
        Self {
            logits: logits.into(),
            appended: Vec::new(),
            pending_sample: None,
            pending_mask: None,
            masked_calls: 0,
            prefill_calls: 0,
            logits_calls: 0,
            choice_calls: 0,
            choice_prefill_calls: 0,
            choice_base_calls: 0,
            choice_append_calls: 0,
            choice_ids: Vec::new(),
            branch_bases: Vec::new(),
            branch_tails: Vec::new(),
            events: Vec::new(),
            fail_append: false,
        }
    }
}

impl TokenExecutor for FakeExecutor {
    type Prefix = Prefix;
    type Prefill = ReadyCompletion<Prefix>;
    type Append = ReadyCompletion<Prefix>;
    type Fork = ReadyCompletion<Prefix>;
    type Scores = ReadyCompletion<Vec<CandidateScore>>;
    type Logits = ReadyCompletion<Vec<f32>>;

    fn prefill(&mut self, input: TokenChunk<'_>) -> ExecutorResult<Self::Prefill> {
        self.prefill_calls += 1;
        self.events.push("prefill");
        self.pending_mask = None;
        Ok(ReadyCompletion::new(Ok(Prefix(input.ids.to_vec()))))
    }

    fn prefill_masked(
        &mut self,
        input: TokenChunk<'_>,
        mask: Rc<[u64]>,
    ) -> ExecutorResult<Self::Prefill> {
        self.masked_calls += 1;
        let result = self.prefill(input);
        self.pending_mask = Some(mask);
        result
    }

    fn append_known(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
    ) -> ExecutorResult<Self::Append> {
        if self.fail_append {
            return Ok(ReadyCompletion::new(Err(ExecutorError::BackendFailure(
                "test append failure",
            ))));
        }
        let mut next = prefix.0.clone();
        next.extend_from_slice(input.ids);
        self.appended.extend_from_slice(input.ids);
        Ok(ReadyCompletion::new(Ok(Prefix(next))))
    }

    fn sampled_token(&mut self, _prefix: &Self::Prefix) -> ExecutorResult<Option<TokenId>> {
        self.logits_calls += 1;
        let logits = self
            .logits
            .pop_front()
            .unwrap_or(Err(ExecutorError::BackendFailure(
                "unexpected logits request",
            )))?;
        let mask = self.pending_mask.take();
        let mut best_id = None;
        let mut best = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().enumerate() {
            if let Some(mask) = &mask {
                let allowed = mask
                    .get(index / 64)
                    .is_some_and(|word| word & (1_u64 << (index % 64)) != 0);
                if !allowed {
                    continue;
                }
            }
            if !value.is_finite() {
                return Err(ExecutorError::BackendFailure("non-finite test logit"));
            }
            if best_id.is_none() || value > best {
                best = value;
                best_id = Some(
                    u32::try_from(index)
                        .map_err(|_| ExecutorError::Overflow("test logit index overflows u32"))?,
                );
            }
        }
        let best_id = best_id.ok_or(ExecutorError::BackendFailure(
            "mask excluded every candidate",
        ))?;
        self.pending_sample = Some(best_id);
        Ok(Some(best_id))
    }

    fn append_argmax(&mut self, prefix: Self::Prefix) -> ExecutorResult<Self::Append> {
        if self.fail_append {
            return Ok(ReadyCompletion::new(Err(ExecutorError::BackendFailure(
                "test append failure",
            ))));
        }
        let token = self
            .pending_sample
            .take()
            .ok_or(ExecutorError::BackendFailure(
                "append_argmax without a resolved sample",
            ))?;
        self.pending_mask = None;
        let mut next = prefix.0.clone();
        next.push(token);
        self.appended.push(token);
        Ok(ReadyCompletion::new(Ok(Prefix(next))))
    }

    fn append_argmax_masked(
        &mut self,
        prefix: Self::Prefix,
        mask: Rc<[u64]>,
    ) -> ExecutorResult<Self::Append> {
        self.masked_calls += 1;
        let result = self.append_argmax(prefix);
        if result.is_ok() {
            self.pending_mask = Some(mask);
        }
        result
    }

    fn fork(&mut self, prefix: &Self::Prefix) -> ExecutorResult<Self::Fork> {
        Ok(ReadyCompletion::new(Ok(prefix.clone())))
    }

    fn next_logits(&mut self, _prefix: &Self::Prefix) -> ExecutorResult<Self::Logits> {
        self.logits_calls += 1;
        let result = self
            .logits
            .pop_front()
            .unwrap_or(Err(ExecutorError::BackendFailure(
                "unexpected logits request",
            )));
        Ok(ReadyCompletion::new(result))
    }

    fn score_candidates(
        &mut self,
        _prefix: &Self::Prefix,
        _candidates: &[&[TokenId]],
    ) -> ExecutorResult<Self::Scores> {
        Ok(ReadyCompletion::new(Ok(Vec::new())))
    }
}

impl TokenChoiceExecutor for FakeExecutor {
    type ChoiceLogits = ReadyCompletion<Vec<f32>>;
    type ChoicePrefill = ReadyCompletion<Vec<f32>>;
    type ChoiceAppend = ReadyCompletion<Vec<f32>>;

    fn choice_logits(
        &mut self,
        _prefix: &Self::Prefix,
        token_ids: &[TokenId],
    ) -> ExecutorResult<Self::ChoiceLogits> {
        self.choice_calls += 1;
        self.events.push("choice");
        self.choice_ids.extend_from_slice(token_ids);
        let result = self
            .logits
            .pop_front()
            .unwrap_or(Err(ExecutorError::BackendFailure(
                "unexpected logits request",
            )));
        Ok(ReadyCompletion::new(result))
    }

    fn prefill_choice_logits(
        &mut self,
        _input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> ExecutorResult<Self::ChoicePrefill> {
        self.choice_prefill_calls += 1;
        self.events.push("choice_prefill");
        self.choice_ids.extend_from_slice(token_ids);
        let result = self
            .logits
            .pop_front()
            .unwrap_or(Err(ExecutorError::BackendFailure(
                "unexpected logits request",
            )));
        Ok(ReadyCompletion::new(result))
    }

    fn prefill_choice_base(&mut self, input: TokenChunk<'_>) -> ExecutorResult<Self::Prefill> {
        self.choice_base_calls += 1;
        self.events.push("choice_base");
        Ok(ReadyCompletion::new(Ok(Prefix(input.ids.to_vec()))))
    }

    fn append_choice_logits(
        &mut self,
        prefix: &Self::Prefix,
        input: TokenChunk<'_>,
        token_ids: &[TokenId],
    ) -> ExecutorResult<Self::ChoiceAppend> {
        self.choice_append_calls += 1;
        self.events.push("choice_append");
        self.choice_ids.extend_from_slice(token_ids);
        self.branch_bases.push(prefix.0.clone());
        self.branch_tails.push(input.ids.to_vec());
        let result = self
            .logits
            .pop_front()
            .unwrap_or(Err(ExecutorError::BackendFailure(
                "unexpected logits request",
            )));
        Ok(ReadyCompletion::new(result))
    }
}

/// Test constraint allowing a fixed id set; records every advanced token.
struct AllowList {
    mask: Rc<[u64]>,
    advanced: Vec<TokenId>,
}

impl AllowList {
    fn new(ids: &[TokenId]) -> Self {
        let mut mask = vec![0_u64; (MODEL_VOCAB_SIZE as usize).div_ceil(64)];
        for &id in ids {
            mask[id as usize / 64] |= 1_u64 << (id as usize % 64);
        }
        Self {
            mask: mask.into(),
            advanced: Vec::new(),
        }
    }
}

impl DecodeConstraint for AllowList {
    fn allowed(&mut self) -> Rc<[u64]> {
        Rc::clone(&self.mask)
    }

    fn advance(&mut self, token: TokenId) {
        self.advanced.push(token);
    }
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_json_bytes(
        include_bytes!("compact_tokenizer.json"),
        TokenizerLimits::default(),
    )
    .unwrap_or_else(|error| panic!("compact tokenizer fixture admission failed: {error}"))
}

fn request<'a>(prompt: &'a str, limit: usize, stops: &'a [TokenId]) -> GenerationRequest<'a> {
    GenerationRequest {
        prompt,
        add_bos: false,
        max_output_tokens: limit,
        max_context_tokens: 32,
        stop_token_ids: stops,
        skip_special_tokens: true,
        policy: GenerationPolicy::Greedy,
    }
}

fn logits(winners: &[(TokenId, f32)]) -> Vec<f32> {
    let mut values = vec![-10.0; MODEL_VOCAB_SIZE as usize];
    for &(id, value) in winners {
        values[id as usize] = value;
    }
    values
}

#[test]
fn emits_plaintext_and_stops_only_for_explicit_eos() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(vec![Ok(logits(&[(2, 1.0)])), Ok(logits(&[(7, 1.0)]))]);
    let mut cancellation = NeverCancel;
    let result = generate(
        &mut executor,
        &tokenizer,
        &request("a", 3, &[7]),
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("generation should succeed: {error}"));

    assert_eq!(result.input_ids, vec![0]);
    assert_eq!(result.generated_ids, vec![2]);
    assert_eq!(result.text, "b");
    assert_eq!(result.stop_reason, StopReason::StopToken(7));
    assert_eq!(executor.appended, vec![2]);
}

#[test]
fn special_tokens_are_not_implicit_stops() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(vec![Ok(logits(&[(7, 1.0)]))]);
    let mut cancellation = NeverCancel;
    let result = generate(
        &mut executor,
        &tokenizer,
        &request("a", 1, &[3]),
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("generation should emit a non-stop special token: {error}"));

    assert_eq!(result.generated_ids, vec![7]);
    assert_eq!(result.text, "");
    assert_eq!(result.stop_reason, StopReason::MaxOutputTokens);
}

#[test]
fn preserves_utf8_token_boundaries() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(vec![
        Ok(logits(&[(5, 1.0)])),
        Ok(logits(&[(6, 1.0)])),
        Ok(logits(&[(7, 1.0)])),
    ]);
    let mut cancellation = NeverCancel;
    let result = generate(
        &mut executor,
        &tokenizer,
        &request("a", 3, &[7]),
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("generation should join split UTF-8: {error}"));

    assert_eq!(result.generated_ids, vec![5, 6]);
    assert_eq!(result.text, "\u{e9}");
}

#[test]
fn zero_output_returns_without_executor_work_and_preserves_bos_choice() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(Vec::new());
    let mut cancellation = NeverCancel;
    let mut generation_request = request("", 0, &[7]);
    generation_request.add_bos = true;
    let result = generate(
        &mut executor,
        &tokenizer,
        &generation_request,
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("zero-output request should return directly: {error}"));

    assert_eq!(result.input_ids, vec![1]);
    assert!(result.generated_ids.is_empty());
    assert_eq!(result.text, "");
    assert_eq!(executor.prefill_calls, 0);
    assert_eq!(executor.logits_calls, 0);
}

#[test]
fn rejected_candidate_ids_and_failed_appends_are_not_published() {
    let tokenizer = tokenizer();
    let mut unmapped = FakeExecutor::new(vec![Ok(logits(&[(64_402, 1.0)]))]);
    let mut cancellation = NeverCancel;
    assert!(matches!(
        generate(
            &mut unmapped,
            &tokenizer,
            &request("a", 1, &[7]),
            &mut cancellation
        ),
        Err(GenerationError::Tokenizer(_))
    ));
    assert!(unmapped.appended.is_empty());

    let mut append_failure = FakeExecutor::new(vec![Ok(logits(&[(2, 1.0)]))]);
    append_failure.fail_append = true;
    assert!(matches!(
        generate(
            &mut append_failure,
            &tokenizer,
            &request("a", 1, &[7]),
            &mut cancellation,
        ),
        Err(GenerationError::Executor(ExecutorError::BackendFailure(
            "test append failure"
        )))
    ));
    assert!(append_failure.appended.is_empty());
}

#[test]
fn cancellation_and_context_limits_are_checked_before_next_token() {
    let tokenizer = tokenizer();
    let mut cancelled = FakeExecutor::new(vec![Ok(logits(&[(2, 1.0)]))]);
    let mut checks = 0_u8;
    let mut cancellation = || {
        checks += 1;
        checks >= 2
    };
    assert_eq!(
        generate(
            &mut cancelled,
            &tokenizer,
            &request("a", 1, &[7]),
            &mut cancellation
        ),
        Err(GenerationError::Cancelled)
    );
    assert!(cancelled.appended.is_empty());

    let mut bounded = FakeExecutor::new(Vec::new());
    let mut no_cancel = NeverCancel;
    let mut tight = request("a", 1, &[7]);
    tight.max_context_tokens = 1;
    assert_eq!(
        generate(&mut bounded, &tokenizer, &tight, &mut no_cancel),
        Err(GenerationError::ContextLimit {
            required: 2,
            limit: 1,
        })
    );
    assert_eq!(bounded.prefill_calls, 1);
    assert_eq!(bounded.logits_calls, 0);
}

#[test]
fn constrained_generate_skips_masked_out_argmax_winners() {
    let tokenizer = tokenizer();
    // Step 1: id 5 has the top logit but is masked out; id 2 wins instead.
    // Step 2: the stop token wins and is allowed, so generation ends.
    let mut executor = FakeExecutor::new(vec![
        Ok(logits(&[(5, 2.0), (2, 1.0)])),
        Ok(logits(&[(7, 1.0)])),
    ]);
    let mut constraint = AllowList::new(&[2, 7]);
    let mut cancellation = NeverCancel;
    let result = generate_constrained(
        &mut executor,
        &tokenizer,
        &request("a", 3, &[7]),
        &mut constraint,
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("constrained generation should succeed: {error}"));

    assert_eq!(result.generated_ids, vec![2]);
    assert_eq!(result.text, "b");
    assert_eq!(result.stop_reason, StopReason::StopToken(7));
    assert_eq!(executor.appended, vec![2]);
    // prefill_masked plus one append_argmax_masked carried the mask.
    assert_eq!(executor.masked_calls, 2);
    assert_eq!(constraint.advanced, vec![2]);
}

#[test]
fn constrained_generate_continues_when_the_stop_token_is_masked_out() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(vec![Ok(logits(&[(2, 1.0)])), Ok(logits(&[(2, 1.0)]))]);
    let mut constraint = AllowList::new(&[2]);
    let mut cancellation = NeverCancel;
    let result = generate_constrained(
        &mut executor,
        &tokenizer,
        &request("a", 2, &[7]),
        &mut constraint,
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("constrained generation should succeed: {error}"));

    assert_eq!(result.generated_ids, vec![2, 2]);
    assert_eq!(result.stop_reason, StopReason::MaxOutputTokens);
    // prefill_masked plus two append_argmax_masked calls.
    assert_eq!(executor.masked_calls, 3);
    assert_eq!(constraint.advanced, vec![2, 2]);
}

#[test]
fn constrained_generate_surfaces_an_empty_candidate_row() {
    let tokenizer = tokenizer();
    let mut executor = FakeExecutor::new(vec![Ok(logits(&[(2, 1.0)]))]);
    let mut constraint = AllowList::new(&[]);
    let mut cancellation = NeverCancel;
    let result = generate_constrained(
        &mut executor,
        &tokenizer,
        &request("a", 1, &[7]),
        &mut constraint,
        &mut cancellation,
    );
    assert_eq!(
        result,
        Err(GenerationError::Executor(ExecutorError::BackendFailure(
            "mask excluded every candidate"
        )))
    );
}

fn criteria<'a>(specs: &'a [(&'a str, &'a str)]) -> Vec<ChoiceCriterion<'a>> {
    specs
        .iter()
        .map(|&(name, tail)| ChoiceCriterion { name, tail })
        .collect()
}

fn choice_request<'a>(base: &'a str, criteria: &'a [ChoiceCriterion<'a>]) -> ChoiceRequest<'a> {
    ChoiceRequest {
        base_prompt: base,
        criteria,
        true_selector: "x",
        false_selector: "b",
        add_bos: false,
        max_context_tokens: 32,
    }
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn choose_scores_each_criterion_serially() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("billing", "x"), ("technical", "ax"), ("general", "xx")]);
    let mut executor = FakeExecutor::new(vec![
        Ok(vec![5.0, 4.0]),
        Ok(vec![9.0, 9.0]),
        Ok(vec![1.5, 0.5]),
    ]);
    let mut cancellation = NeverCancel;
    let result = choose(
        &mut executor,
        &tokenizer,
        &choice_request("a", &criteria),
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("choice should succeed: {error}"));

    assert_eq!(executor.prefill_calls, 0);
    assert_eq!(executor.choice_calls, 0);
    assert_eq!(executor.choice_prefill_calls, 0);
    assert_eq!(executor.choice_base_calls, 1);
    assert_eq!(executor.choice_append_calls, 3);
    assert_eq!(executor.choice_ids, vec![4, 2, 4, 2, 4, 2]);
    assert_eq!(
        executor.events,
        [
            "choice_base",
            "choice_append",
            "choice_append",
            "choice_append"
        ]
    );
    assert_eq!(executor.branch_bases, [vec![0], vec![0], vec![0]]);
    assert_eq!(executor.branch_tails, [vec![4], vec![0, 4], vec![4, 4]]);
    assert_eq!(executor.logits_calls, 0);
    assert_eq!(executor.masked_calls, 0);
    assert!(executor.appended.is_empty());

    let weight = 1.0_f64.exp();
    let denominator = 2.0 * weight + 1.0;
    assert_eq!(
        result.probabilities,
        vec![
            ChoiceProbability {
                name: "billing".to_owned(),
                probability: (weight / denominator) as f32,
            },
            ChoiceProbability {
                name: "technical".to_owned(),
                probability: (1.0 / denominator) as f32,
            },
            ChoiceProbability {
                name: "general".to_owned(),
                probability: (weight / denominator) as f32,
            },
        ]
    );
    assert_eq!(result.choice, "billing");
    let sum: f32 = result
        .probabilities
        .iter()
        .map(|probability| probability.probability)
        .sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn choose_prefers_the_first_criterion_on_an_exact_evidence_tie() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("billing", "x"), ("technical", "ax")]);
    let mut executor = FakeExecutor::new(vec![Ok(vec![3.0, 1.0]), Ok(vec![7.0, 5.0])]);
    let mut cancellation = NeverCancel;
    let result = choose(
        &mut executor,
        &tokenizer,
        &choice_request("a", &criteria),
        &mut cancellation,
    )
    .unwrap_or_else(|error| panic!("tied choice should succeed: {error}"));

    assert_eq!(result.choice, "billing");
}

#[test]
#[allow(clippy::float_cmp)]
fn choose_confidence_is_zero_for_uniform_and_one_for_one_hot() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("a", "x"), ("b", "ax"), ("c", "xx")]);
    let request = choice_request("a", &criteria);

    let mut uniform = FakeExecutor::new(vec![
        Ok(vec![2.0, 2.0]),
        Ok(vec![5.0, 5.0]),
        Ok(vec![9.0, 9.0]),
    ]);
    let mut cancellation = NeverCancel;
    let result = choose(&mut uniform, &tokenizer, &request, &mut cancellation)
        .unwrap_or_else(|error| panic!("uniform choice should succeed: {error}"));
    assert_eq!(result.confidence, 0.0);

    let mut dominant = FakeExecutor::new(vec![
        Ok(vec![800.0, 0.0]),
        Ok(vec![0.0, 0.0]),
        Ok(vec![0.0, 0.0]),
    ]);
    let mut cancellation = NeverCancel;
    let result = choose(&mut dominant, &tokenizer, &request, &mut cancellation)
        .unwrap_or_else(|error| panic!("dominant choice should succeed: {error}"));
    assert_eq!(result.confidence, 1.0);
    assert_eq!(result.choice, "a");
}

#[test]
fn finish_choice_is_invariant_to_per_criterion_additive_shifts() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("billing", "x"), ("technical", "ax"), ("general", "xx")]);
    let request = choice_request("a", &criteria);

    let prepared = prepare_choice(&tokenizer, &request)
        .unwrap_or_else(|error| panic!("criteria should prepare: {error}"));
    let unshifted = finish_choice(
        prepared,
        vec![vec![2.0, 1.0], vec![0.0, 1.0], vec![0.5, 0.5]],
    )
    .unwrap_or_else(|error| panic!("unshifted choice should finish: {error}"));

    let prepared = prepare_choice(&tokenizer, &request)
        .unwrap_or_else(|error| panic!("criteria should prepare: {error}"));
    let shifted = finish_choice(
        prepared,
        vec![vec![2.0, 1.0], vec![4.0, 5.0], vec![0.5, 0.5]],
    )
    .unwrap_or_else(|error| panic!("shifted choice should finish: {error}"));

    assert_eq!(unshifted, shifted);
}

#[test]
fn prepare_choice_validates_criteria_and_selectors() {
    let tokenizer = tokenizer();
    let valid = criteria(&[("billing", "x"), ("technical", "ax")]);
    let prepared = prepare_choice(&tokenizer, &choice_request("a", &valid))
        .unwrap_or_else(|error| panic!("valid criteria should prepare: {error}"));
    assert_eq!(prepared.token_ids(), &[4, 2]);
    assert_eq!(prepared.base_input_ids(), &[0]);
    assert_eq!(prepared.criteria().len(), 2);
    assert_eq!(prepared.criteria()[0].name(), "billing");
    assert_eq!(prepared.criteria()[0].tail_ids(), &[4]);
    assert_eq!(prepared.criteria()[1].tail_ids(), &[0, 4]);

    let mut bos = choice_request("a", &valid);
    bos.add_bos = true;
    let prepared = prepare_choice(&tokenizer, &bos)
        .unwrap_or_else(|error| panic!("BOS request should prepare: {error}"));
    assert_eq!(prepared.base_input_ids(), &[1, 0]);

    for specs in [
        &[("billing", "x")][..],
        &[("", "x"), ("technical", "ax")][..],
        &[("dup", "x"), ("dup", "ax")][..],
        &[("billing", ""), ("technical", "ax")][..],
    ] {
        assert!(
            matches!(
                prepare_choice(&tokenizer, &choice_request("a", &criteria(specs))),
                Err(ChoiceError::InvalidChoices(_))
            ),
            "criteria {specs:?} should be rejected"
        );
    }

    assert!(matches!(
        prepare_choice(&tokenizer, &choice_request("", &valid)),
        Err(ChoiceError::InvalidChoices(_))
    ));

    let mut empty_true = choice_request("a", &valid);
    empty_true.true_selector = "";
    assert!(matches!(
        prepare_choice(&tokenizer, &empty_true),
        Err(ChoiceError::InvalidChoices(_))
    ));

    let mut same_selector = choice_request("a", &valid);
    same_selector.false_selector = "x";
    assert!(matches!(
        prepare_choice(&tokenizer, &same_selector),
        Err(ChoiceError::InvalidChoices(_))
    ));

    let mut multi_token = choice_request("a", &valid);
    multi_token.true_selector = "ax";
    assert!(matches!(
        prepare_choice(&tokenizer, &multi_token),
        Err(ChoiceError::InvalidChoices(_))
    ));

    // "a" + "b" merges to the single token "ab", so a "b" tail cannot
    // compose after base "a".
    let merged_tail = criteria(&[("billing", "b"), ("technical", "x")]);
    assert!(matches!(
        prepare_choice(&tokenizer, &choice_request("a", &merged_tail)),
        Err(ChoiceError::InvalidChoices(_))
    ));

    // "x" + "a" composes, but the false selector "b" then merges into "ab".
    let merged_selector = criteria(&[("billing", "a"), ("technical", "x")]);
    assert!(matches!(
        prepare_choice(&tokenizer, &choice_request("x", &merged_selector)),
        Err(ChoiceError::InvalidChoices(_))
    ));

    let mut tight = choice_request("a", &valid);
    tight.max_context_tokens = 1;
    assert!(matches!(
        prepare_choice(&tokenizer, &tight),
        Err(ChoiceError::ContextLimit {
            required: 2,
            limit: 1
        })
    ));
}

#[test]
fn prepare_choice_rejects_more_than_255_criteria() {
    let tokenizer = tokenizer();
    let names: Vec<String> = (0..256).map(|index| format!("criterion{index}")).collect();
    let many: Vec<ChoiceCriterion<'_>> = names
        .iter()
        .map(|name| ChoiceCriterion {
            name: name.as_str(),
            tail: "x",
        })
        .collect();
    assert!(matches!(
        prepare_choice(&tokenizer, &choice_request("a", &many)),
        Err(ChoiceError::InvalidChoices(
            "choice supports at most 255 criteria"
        ))
    ));
}

#[test]
fn finish_choice_requires_one_finite_pair_per_criterion() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("billing", "x"), ("technical", "ax")]);
    let request = choice_request("a", &criteria);

    let prepared = prepare_choice(&tokenizer, &request)
        .unwrap_or_else(|error| panic!("criteria should prepare: {error}"));
    assert!(matches!(
        finish_choice(prepared, vec![vec![0.5, 0.5]]),
        Err(ChoiceError::Executor(ExecutorError::InvalidShape(_)))
    ));

    let prepared = prepare_choice(&tokenizer, &request)
        .unwrap_or_else(|error| panic!("criteria should prepare: {error}"));
    assert!(matches!(
        finish_choice(prepared, vec![vec![0.5, 0.5, 0.5], vec![0.0, 0.0]]),
        Err(ChoiceError::Executor(ExecutorError::InvalidShape(_)))
    ));

    let prepared = prepare_choice(&tokenizer, &request)
        .unwrap_or_else(|error| panic!("criteria should prepare: {error}"));
    assert_eq!(
        finish_choice(prepared, vec![vec![0.5, 0.5], vec![f32::NAN, 0.0]]),
        Err(ChoiceError::NonFiniteLogit { choice_index: 1 })
    );
}

#[test]
fn choose_propagates_executor_failure_and_late_cancellation() {
    let tokenizer = tokenizer();
    let criteria = criteria(&[("billing", "x"), ("technical", "ax"), ("general", "xx")]);

    let mut failing = FakeExecutor::new(vec![
        Ok(vec![1.0, 0.0]),
        Err(ExecutorError::BackendFailure("test choice failure")),
    ]);
    let mut cancellation = NeverCancel;
    assert_eq!(
        choose(
            &mut failing,
            &tokenizer,
            &choice_request("a", &criteria),
            &mut cancellation
        ),
        Err(ChoiceError::Executor(ExecutorError::BackendFailure(
            "test choice failure"
        )))
    );
    assert_eq!(failing.prefill_calls, 0);
    assert_eq!(failing.choice_calls, 0);
    assert_eq!(failing.choice_prefill_calls, 0);
    assert_eq!(failing.choice_base_calls, 1);
    assert_eq!(failing.choice_append_calls, 2);

    let mut executor = FakeExecutor::new(vec![Ok(vec![1.0, 0.0]); 3]);
    let mut checks = 0_u8;
    let mut cancelling = || {
        checks += 1;
        checks >= 3
    };
    assert_eq!(
        choose(
            &mut executor,
            &tokenizer,
            &choice_request("a", &criteria),
            &mut cancelling
        ),
        Err(ChoiceError::Cancelled)
    );
    assert_eq!(executor.prefill_calls, 0);
    assert_eq!(executor.choice_calls, 0);
    assert_eq!(executor.choice_prefill_calls, 0);
    assert_eq!(executor.choice_base_calls, 1);
    assert_eq!(executor.choice_append_calls, 2);
    assert!(executor.appended.is_empty());
}

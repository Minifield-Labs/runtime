use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use minifield_engine_api::{DecodeConstraint, TokenId};

/// Retain at most 128 recent state masks. At 65,536 tokens this holds about
/// 1 MiB of mask data, plus state keys and map overhead.
const DEFAULT_MASK_CACHE_CAP: usize = 128;

/// A byte-level document acceptor driven by [`Enforcer`].
///
/// `feed` mutates observable state; `accepts` must leave it unchanged
/// (implementations may use internal scratch space).
pub trait Machine {
    /// Feed one byte; returns false when the byte is invalid here.
    fn feed(&mut self, byte: u8) -> bool;
    /// Would feeding `bytes` succeed from the current state?
    fn accepts(&mut self, bytes: &[u8]) -> bool;
    /// The document is complete (or completable) in this state.
    fn complete(&self) -> bool;
    /// A terminal document that should stop generation immediately.
    fn finished(&self) -> bool {
        false
    }
    /// Append a byte-identity of the state for mask caching.
    fn key(&self, out: &mut Vec<u8>);
}

/// Grammar-constrained decode: maintains a byte-level acceptor over
/// emitted tokens and yields allowed-id bitsets. Its FIFO cache retains at
/// most 128 state masks; evicted states are recomputed when visited again.
pub struct Enforcer<M> {
    /// Raw byte string per token id; entries may be empty for special or
    /// unmapped ids, which are never allowed.
    vocab: Vec<Vec<u8>>,
    machine: M,
    cache: HashMap<Vec<u8>, Rc<[u64]>>,
    cache_order: VecDeque<Vec<u8>>,
    /// Stop token allowed once the document is complete.
    eos: TokenId,
}

impl<M: Machine> Enforcer<M> {
    /// `vocab[id]` must be the token's raw bytes; pass an empty vec for ids
    /// that have no byte form. `eos` becomes allowed only once the machine
    /// reports a complete document; its bytes never enter the machine.
    #[must_use]
    pub fn with_machine(vocab: Vec<Vec<u8>>, eos: TokenId, machine: M) -> Self {
        Self {
            vocab,
            machine,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            eos,
        }
    }

    /// The machine's document is complete or completable.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.machine.complete()
    }
}

impl<M: Machine> DecodeConstraint for Enforcer<M> {
    fn finished(&self) -> bool {
        self.machine.finished()
    }

    fn allowed(&mut self) -> Rc<[u64]> {
        let mut key = Vec::new();
        self.machine.key(&mut key);
        if let Some(mask) = self.cache.get(&key) {
            return Rc::clone(mask);
        }
        // Gate the vocab scan on each token's first byte: only a handful of
        // bytes can open a valid continuation from this state, so the full
        // simulation runs just for those tokens.
        let mut first = [false; 256];
        for byte in 0..=255_u8 {
            if self.machine.accepts(&[byte]) {
                first[usize::from(byte)] = true;
            }
        }
        let eos = usize::try_from(self.eos).unwrap_or(usize::MAX);
        let mut mask = vec![0_u64; self.vocab.len().div_ceil(64)];
        for (id, word) in mask.iter_mut().enumerate() {
            for bit in 0..64 {
                let token = id * 64 + bit;
                if token < self.vocab.len()
                    && token != eos
                    && !self.vocab[token].is_empty()
                    && first[usize::from(self.vocab[token][0])]
                    && self.machine.accepts(&self.vocab[token])
                {
                    *word |= 1_u64 << bit;
                }
            }
        }
        if self.machine.complete()
            && eos < self.vocab.len()
            && let Some(word) = mask.get_mut(eos / 64)
        {
            *word |= 1_u64 << (eos % 64);
        }
        let mask: Rc<[u64]> = mask.into();
        if self.cache.len() == DEFAULT_MASK_CACHE_CAP
            && let Some(oldest) = self.cache_order.pop_front()
        {
            self.cache.remove(&oldest);
        }
        self.cache_order.push_back(key.clone());
        self.cache.insert(key, Rc::clone(&mask));
        mask
    }

    fn advance(&mut self, token: TokenId) {
        if token == self.eos {
            return;
        }
        let Some(bytes) = usize::try_from(token)
            .ok()
            .and_then(|index| self.vocab.get(index))
        else {
            return;
        };
        for &byte in bytes {
            // The executor's masked argmax only emits allowed ids; a feed
            // failure here would mean the mask and machine disagree.
            let _ = self.machine.feed(byte);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use minifield_engine_api::DecodeConstraint;

    use super::{DEFAULT_MASK_CACHE_CAP, Enforcer, Machine};

    /// A counter gives each visit a distinct state while accepting the
    /// same token, so eviction can be observed independently of grammar.
    struct CounterMachine {
        state: usize,
    }

    impl Machine for CounterMachine {
        fn feed(&mut self, byte: u8) -> bool {
            if byte != b'x' {
                return false;
            }
            self.state += 1;
            true
        }

        fn accepts(&mut self, bytes: &[u8]) -> bool {
            bytes.iter().all(|&byte| byte == b'x')
        }

        fn complete(&self) -> bool {
            false
        }

        fn key(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&self.state.to_le_bytes());
        }
    }

    #[test]
    fn cache_evicts_oldest_mask_and_recomputes_it_without_changing_bits() {
        let mut enforcer = Enforcer::with_machine(
            vec![Vec::new(), b"x".to_vec()],
            0,
            CounterMachine { state: 0 },
        );
        let first = enforcer.allowed();
        assert_eq!(first.as_ref(), &[2]);
        assert!(Rc::ptr_eq(&first, &enforcer.allowed()));
        for _ in 0..DEFAULT_MASK_CACHE_CAP {
            enforcer.advance(1);
            let latest = enforcer.allowed();
            assert!(Rc::ptr_eq(&latest, &enforcer.allowed()));
            assert!(enforcer.cache.len() <= DEFAULT_MASK_CACHE_CAP);
        }
        assert_eq!(enforcer.cache.len(), DEFAULT_MASK_CACHE_CAP);
        assert_eq!(enforcer.cache_order.len(), DEFAULT_MASK_CACHE_CAP);
        enforcer.machine.state = 0;
        let recomputed = enforcer.allowed();
        assert_eq!(first, recomputed);
        // Holding the original Rc proves this is a fresh computation,
        // rather than a cache entry retained beyond the bound.
        assert!(!Rc::ptr_eq(&first, &recomputed));
        assert!(Rc::ptr_eq(&recomputed, &enforcer.allowed()));
        assert_eq!(enforcer.cache.len(), DEFAULT_MASK_CACHE_CAP);
        assert_eq!(enforcer.cache_order.len(), DEFAULT_MASK_CACHE_CAP);
    }
}

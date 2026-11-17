//! Validation-loss-based early stopping.

pub(super) const DEFAULT_PATIENCE: usize = 3;
pub(super) const DEFAULT_MIN_DELTA: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EarlyStoppingDecision {
    Improved,
    Continue,
    Stop,
}

#[derive(Debug)]
pub(super) struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    best_loss: Option<f32>,
    bad_epochs: usize,
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self {
            patience: DEFAULT_PATIENCE,
            min_delta: DEFAULT_MIN_DELTA,
            best_loss: None,
            bad_epochs: 0,
        }
    }
}

impl EarlyStopping {
    pub(super) fn observe(&mut self, loss: f32) -> EarlyStoppingDecision {
        let improved = match self.best_loss {
            None => true,
            Some(best) => loss < best - self.min_delta,
        };

        if improved {
            self.best_loss = Some(loss);
            self.bad_epochs = 0;
            EarlyStoppingDecision::Improved
        } else {
            self.bad_epochs = self.bad_epochs.saturating_add(1);

            if self.bad_epochs >= self.patience {
                EarlyStoppingDecision::Stop
            } else {
                EarlyStoppingDecision::Continue
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_observation_is_an_improvement() {
        let mut stopping = EarlyStopping::default();
        assert_eq!(stopping.observe(5.0), EarlyStoppingDecision::Improved);
    }

    #[test]
    fn small_changes_do_not_count_as_improvements() {
        let mut stopping = EarlyStopping::default();
        assert_eq!(stopping.observe(5.0), EarlyStoppingDecision::Improved);
        assert_eq!(stopping.observe(4.995), EarlyStoppingDecision::Continue);
    }

    #[test]
    fn stops_after_three_non_improving_epochs() {
        let mut stopping = EarlyStopping::default();
        stopping.observe(5.0);
        assert_eq!(stopping.observe(5.1), EarlyStoppingDecision::Continue);
        assert_eq!(stopping.observe(5.2), EarlyStoppingDecision::Continue);
        assert_eq!(stopping.observe(5.3), EarlyStoppingDecision::Stop);
    }

    #[test]
    fn improvement_resets_patience() {
        let mut stopping = EarlyStopping::default();
        stopping.observe(5.0);
        stopping.observe(5.1);
        assert_eq!(stopping.observe(4.98), EarlyStoppingDecision::Improved);
        assert_eq!(stopping.observe(5.0), EarlyStoppingDecision::Continue);
    }
}

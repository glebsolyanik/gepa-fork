"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable, Any, Callable

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx


@runtime_checkable
class EvaluationPolicy(Protocol[DataId, DataInst]):  # type: ignore
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances."""

    @abstractmethod
    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Select examples for evaluation for a program"""
        ...

    @abstractmethod
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        ...

    @abstractmethod
    def compute_metric(
        self, 
        predictions: dict[DataId, Any], 
        ground_truth: dict[DataId, Any]
    ) -> float:
        """Вычислить метрику на основе predictions и ground truth"""
        ...



class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        predictions = state.prog_candidate_val_predictions[program_idx]
        ground_truth = state.prog_candidate_val_ground_truth[program_idx]

        if predictions and ground_truth:
            return self.compute_metric(predictions, ground_truth)
        else:
            return state.get_program_average_val_subset(program_idx)[0]

    def compute_metric(
        self,
        predictions: dict[DataId, Any],
        ground_truth: dict[DataId, Any],
    ) -> float:
        """Вычисляет среднее (для обратной совместимости)."""
        if not predictions:
            return float("-inf")
        pred_values = list(predictions.values())
        if pred_values and isinstance(pred_values[0], (int, float)):
            scores = pred_values
            return sum(scores) / len(scores)
        return float("-inf")

class F2EvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that uses F2 score for optimization."""
    
    def __init__(self, prediction_extractor: Callable[[Any], int] | None = None):
        """
        Args:
            prediction_extractor: Функция для извлечения бинарного prediction из RolloutOutput.
                                 Если None, предполагается что predictions уже бинарные (0/1).
        """
        self.prediction_extractor = prediction_extractor

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program with the highest F2 score."""
        best_idx, best_f2 = -1, float("-inf")
        
        for program_idx in range(len(state.program_candidates)):
            f2_score = self.get_valset_score(program_idx, state)
            if f2_score > best_f2:
                best_f2 = f2_score
                best_idx = program_idx
        
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the F2 score of the program on the valset"""
        predictions = state.prog_candidate_val_predictions[program_idx]
        ground_truth = state.prog_candidate_val_ground_truth[program_idx]
        
        if predictions and ground_truth:
            return self.compute_metric(predictions, ground_truth)
        else:
            return state.get_program_average_val_subset(program_idx)[0]
    
    
    def compute_metric(
        self,
        predictions: dict[DataId, Any],
        ground_truth: dict[DataId, Any],
    ) -> float:
        """Вычисляет F2 score."""
        if not predictions or not ground_truth:
            return float("-inf")
        
        all_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
        if not all_ids:
            return float("-inf")
        
        y_pred = []
        y_true = []
        
        for val_id in all_ids:
            pred = predictions[val_id]
            gt = ground_truth[val_id]
            
            if self.prediction_extractor:
                y_pred.append(self.prediction_extractor(pred))
                y_true.append(self.prediction_extractor(gt))
            else:
                y_pred.append(int(pred) if isinstance(pred, (int, float, bool)) else 0)
                y_true.append(int(gt) if isinstance(gt, (int, float, bool)) else 0)
        
        from sklearn.metrics import fbeta_score
        return fbeta_score(y_true, y_pred, beta=2)


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "F2EvaluationPolicy",
]
